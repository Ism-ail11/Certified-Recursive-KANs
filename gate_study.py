#!/usr/bin/env python3
"""Fit only the gate, calibrate it on validation, and compare paired test predictions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import median
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from sklearn.metrics import f1_score

from contextual_gain import ContextualGain, MODES, STRENGTHS, contexts, fit_gain, scores_at
from data import DATASETS, SEEDS, load_split
from fit_baselines import load_model
from splines import loss_gradient


METHODS = (
    ("standard", "Standard cubic KAN"),
    ("fixed", "Fixed-gain anchor"),
    ("reciprocal", "Original reciprocal gate"),
    ("constant", "Learned constant gains"),
    ("directional", "Contextual reciprocal gains"),
    ("signed_full", "Contextual signed gate, full strength"),
    ("signed", "Calibrated contextual signed gate"),
)
TITLES = dict(METHODS)
TOLERANCE = 1e-5


def chunks(x, batch=256):
    return (x[i:i + batch] for i in range(0, len(x), batch))


def csv_write(path, rows):
    if not rows:
        return
    order = list(rows[0]) + sorted(set().union(*(row.keys() for row in rows)) - set(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=order)
        writer.writeheader()
        writer.writerows(rows)


def save_gate(path, gate, strength, record, anchor):
    data = {f"p{i:02d}": p for i, p in enumerate(gate.snapshot())}
    data.update(center=gate.center, spread=gate.spread, strength=strength, mode=np.asarray(gate.mode),
                pairs=gate.pairs, anchor=np.asarray(anchor))
    np.savez_compressed(path, **data)
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def load_gate(path, anchor=None):
    with np.load(path, allow_pickle=False) as data:
        if anchor is not None and str(data["anchor"]) != anchor:
            raise ValueError("Gate checkpoint belongs to a different frozen anchor")
        gate = ContextualGain(len(data["center"]), int(data["pairs"]), str(data["mode"]), 0)
        gate.center[:], gate.spread[:] = data["center"], data["spread"]
        gate.restore([data[name] for name in sorted(data.files) if name.startswith("p") and name != "pairs"])
        return gate, float(data["strength"])


def anchor_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fit_one(job):
    baseline_root, output, dataset, seed, epochs = job
    d = load_split(dataset, seed)
    source = baseline_root / dataset / "fixed" / f"{seed}.npz"
    model, _ = load_model(source)
    frozen = model.snapshot()
    h = np.concatenate([model.encode(c) for c in chunks(d.x)])
    nominal = model.readout.forward(h)
    context = contexts(d.x, h)
    local = output / dataset / str(seed)
    local.mkdir(parents=True, exist_ok=True)
    digest = anchor_hash(source)
    chosen = {}
    for mode in MODES:
        gate, alpha, history, vals = fit_gain(model, h, nominal, context, d.y, d.train, d.valid,
                                             d.task, mode, seed, epochs)
        record = {"dataset": dataset, "seed": seed, "mode": mode, "selected_strength": alpha,
                  "history": history, "validation_curve": vals, "parameters": gate.parameter_count}
        save_gate(local / f"{mode}.npz", gate, alpha, record, digest)
        chosen[mode] = alpha
    for old, (current, _) in zip(frozen, model.parameters()):
        np.testing.assert_array_equal(old, current)
    return dataset, seed, chosen


def predict(model, x, gate=None, alpha=0., batch=128):
    scores, err, q, fallback, states = [], [], [], [], []
    for sample in chunks(x, batch):
        h = model.encode(sample)
        nominal = model.readout.forward(h)
        scales = gate.forward(contexts(sample, h), alpha) if gate is not None and alpha > 0 else None
        p, result, z = model.from_features(h, nominal, scales, certify=True)
        scores.append(p)
        if result is not None:
            low, high = model.output_bounds(p, h, result.state, result.error, nominal)
            if not ((low <= p).all() and (p <= high).all()):
                raise AssertionError("Invalid outward output interval")
            err.append(result.error.max(axis=1))
            q.append(result.q)
            fallback.append(result.fallbacks)
            states.append(z)
    if not err:
        return np.concatenate(scores), None, None, None, None
    return (np.concatenate(scores), np.concatenate(err), np.concatenate(q),
            np.concatenate(fallback), np.concatenate(states))


def metric(split, y, scores):
    if split.task == "classification":
        decision = scores.argmax(axis=1)
        ok = (decision == y).astype(np.float64)
        return {"accuracy": float(ok.mean()), "macro_f1": float(f1_score(y, decision, average="macro")),
                "nll": loss_gradient(scores, y, split.task)[0]}, decision, 1 - ok
    decision = scores[:, 0] * split.target_scale + split.target_mean
    truth = y * split.target_scale + split.target_mean
    residual = truth - decision
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual)))}, decision, residual ** 2


def evaluate(baseline_root, output, datasets, seeds, repeats):
    table, curves = [], []
    for dataset in datasets:
        for seed in seeds:
            d = load_split(dataset, seed)
            local = output / dataset / str(seed)
            anchor = baseline_root / dataset / "fixed" / f"{seed}.npz"
            base, _ = load_model(anchor)
            std, _ = load_model(baseline_root / dataset / "standard" / f"{seed}.npz")
            old, _ = load_model(baseline_root / dataset / "reciprocal" / f"{seed}.npz")
            digest = anchor_hash(anchor)
            gates = {mode: load_gate(local / f"{mode}.npz", digest) for mode in MODES}
            signed, picked = gates["signed"]
            x, target = d.x[d.test], d.y[d.test]
            panel = {
                "test_index": d.test, "group": d.groups[d.test], "target": target,
                "target_scale": np.array(d.target_scale), "target_mean": np.array(d.target_mean),
            }
            times_x = x[np.linspace(0, len(x) - 1, min(1024, len(x)), dtype=int)]
            choices = [(std, None, 0.), (base, None, 0.), (old, None, 1.),
                       (base, *gates["constant"]), (base, *gates["directional"]),
                       (base, signed, 1.), (base, signed, picked)]
            for (method, _), (model, gate, strength) in zip(METHODS, choices):
                scores, error, q, fallback, _ = predict(model, x, gate, strength)
                info, label, loss = metric(d, target, scores)
                panel[f"prediction_{method}"] = label
                panel[f"loss_{method}"] = loss
                measured = []
                predict(model, times_x, gate, strength)
                for _ in range(repeats):
                    before = perf_counter()
                    predict(model, times_x, gate, strength)
                    measured.append(1000 * (perf_counter() - before) / len(times_x))
                row = dict(dataset=dataset, seed=seed, method=method, n_test=len(x), strength=strength,
                           parameters=model.parameter_count + (gate.parameter_count if gate is not None else 0),
                           ms_per_example=median(measured), **info)
                if error is not None:
                    row.update(certified_fraction=float(np.mean(error <= TOLERANCE)),
                               state_error_p95=float(np.quantile(error, .95)), loop_factor_p95=float(np.quantile(q, .95)),
                               breakpoint_fallbacks_mean=float(np.mean(fallback)))
                table.append(row)
            h = np.concatenate([base.encode(c) for c in chunks(x)])
            nominal = base.readout.forward(h)
            context = contexts(x, h)
            all_unit = scores_at(base, h, nominal, context, signed, 0.)
            default = np.concatenate([base.from_features(hh, n)[0]
                                      for hh, n in zip(chunks(h), chunks(nominal))])
            np.testing.assert_array_equal(all_unit, default)
            for mode, (gate, alpha) in gates.items():
                record = json.loads((local / f"{mode}.json").read_text())
                for strength in STRENGTHS:
                    prediction = scores_at(base, h, nominal, context, gate, strength)
                    info, _, _ = metric(d, target, prediction)
                    raw_strength = str(strength)
                    curves.append(dict(dataset=dataset, seed=seed, mode=mode, strength=strength,
                                       selected=strength == alpha, validation_loss=record["validation_curve"][raw_strength],
                                       test_loss=loss_gradient(prediction, target, d.task)[0], **info))
            scales = signed.forward(context, picked)
            panel["gate_activity"] = np.mean(np.abs(np.log(scales[:, 0::2]))
                                              + np.abs(scales[:, 0::2] * scales[:, 1::2] - 1), axis=1)
            panel["reverse_fraction"] = np.mean(scales[:, 1::2] < 0, axis=1)
            np.savez_compressed(local / "paired_predictions.npz", **panel)
            print(f"Measured {dataset} seed={seed}, signed strength={picked:g}", flush=True)
    csv_write(output / "real_per_seed.csv", table)
    csv_write(output / "strength_curves.csv", curves)
    (output / "manifest.json").write_text(json.dumps({
        "datasets": list(datasets), "seeds": list(seeds), "evaluations": len(table), "latency_repeats": repeats,
        "max_timed_examples": 1024, "batch_size": 128, "hidden_state_tolerance": TOLERANCE,
        "selection": "Validation only. Smallest strength within 0.2% of fixed-anchor loss of the validation minimum.",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--baselines", type=Path, default=Path("results/baselines"))
    parser.add_argument("--output", type=Path, default=Path("results/gating"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    a = parser.parse_args()
    if min(a.epochs, a.workers, a.repeats) < 1:
        parser.error("Use positive experiment settings")
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "evaluate":
        evaluate(a.baselines, a.output, a.datasets, a.seeds, a.repeats)
        return
    jobs = [(a.baselines, a.output, d, seed, a.epochs) for d in a.datasets for seed in a.seeds]
    if a.workers == 1:
        for job in jobs:
            print(*fit_one(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as pool:
            for result in as_completed([pool.submit(fit_one, j) for j in jobs]):
                print(*result.result(), flush=True)


if __name__ == "__main__":
    main()
