#!/usr/bin/env python3
"""Test learned gating on known sign-changing and fixed-gain feedback systems."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from contextual_gain import MODES, fit_gain, scores_at
from gate_study import csv_write, save_gate
from paired_kan import PairedBlock


class FrozenSystem:
    def __init__(self):
        rng = np.random.default_rng(4337)
        self.block = PairedBlock(12, rng)
        self.weight = np.eye(12)
        self.bias = np.zeros(12)
        for stage in (self.block.early, self.block.late):
            grid = stage.grid
            stage.coef[:, 0] = 1.1 * grid + .070 * np.sin(2 * grid)
            indices = np.arange(stage.count)
            polarity = np.where(indices % 2 == 0, 1., -1.)
            stage.coef[:, 1] = polarity[:, None] * (.43 * grid + .035 * np.sin(2 * grid))
            stage.project()

    def from_features(self, h, base, scales):
        return self.block.forward(h, scales), None, None


def teacher_gains(regime, h):
    i = np.arange(h.shape[1] // 2)[None]
    c = np.where(i % 2 == 0, 1., -1.)
    u = .88 * np.tanh(c * regime[:, 0:1] + .45 * regime[:, 1:2] + .18 * h[:, 0::2])
    tau = np.tanh(2.2 * regime[:, 0:1] - c * 1.05 * regime[:, 1:2] + (i % 3 - 1) * .18)
    return np.stack([np.exp(u), np.exp(-u) * tau], axis=2).reshape(h.shape)


def iterate(stage, bias, scales, steps=160):
    p = np.tile(np.arange(stage.count), len(bias))
    a, b = bias.reshape(-1, 2).T
    s, t = scales.reshape(-1, 2).T
    x, y = a.copy(), b.copy()
    for _ in range(steps):
        x, y = a + s * stage._spline(0, p, y), b + t * stage._spline(1, p, x)
    return np.column_stack([x, y]).reshape(bias.shape)


def target_from_iteration(model, h, factors):
    a = model.block.split
    early = iterate(model.block.early, h[:, :a], factors[:, :a])
    late = iterate(model.block.late, h[:, a:] + model.block.cross.forward(early), factors[:, a:])
    return np.concatenate([early, late], axis=1)


def case_inputs():
    rng = np.random.default_rng(29904)
    h = rng.uniform(-1.8, 1.8, (8192, 12))
    regime = rng.uniform(-2., 2., (len(h), 2))
    return h, regime, np.column_stack([regime, h])


def fit_case(job):
    name, path, seed, epochs = job
    h, regime, ctx = case_inputs()
    model = FrozenSystem()
    factors = teacher_gains(regime, h) if name == "switching" else np.ones_like(h)
    y = target_from_iteration(model, h, factors)
    stable = model.block.certify(h, factors)
    if not (np.abs(stable.state - y) <= stable.error + 5e-13).all():
        raise AssertionError("Independent iteration escaped the teacher enclosure")
    order = np.random.default_rng(seed + 579).permutation(len(h))
    train, valid, test = np.split(order, [5120, 6144])
    raw = model.block.forward(h[test], np.ones_like(h[test]))
    rows = [dict(scenario=name, seed=seed, method="fixed", strength=0.,
                 rmse=float(np.sqrt(np.mean((raw - y[test]) ** 2))), n_test=len(test))]
    for mode in MODES:
        gate, selected, history, curve = fit_gain(model, h, h, ctx, y, train, valid, "regression", mode, seed, epochs)
        gate_file = path / f"{name}_{mode}_{seed}.npz"
        save_gate(gate_file, gate, selected, {"history": history, "validation_curve": curve,
                                             "strength": selected, "seed": seed}, "FrozenSystem")
        for variant, strength in ((mode, selected), (f"{mode}_full", 1.)) if mode == "signed" else ((mode, selected),):
            p = scores_at(model, h[test], h[test], ctx[test], gate, strength)
            learned = gate.forward(ctx[test], strength)
            proof = model.block.certify(h[test], learned)
            signs = np.mean((learned[:, 1::2] < 0) == (factors[test, 1::2] < 0))
            rows.append(dict(scenario=name, seed=seed, method=variant, strength=strength,
                             rmse=float(np.sqrt(np.mean((p - y[test]) ** 2))),
                             n_test=len(test), sign_agreement=float(signs),
                             certified_fraction=float(np.mean(proof.error.max(axis=1) <= 1e-5))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/controlled"))
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args()
    if min(a.epochs, a.workers) < 1:
        parser.error("Use positive counts")
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [(name, a.output, seed, a.epochs) for name in ("switching", "constant") for seed in (17, 29, 43)]
    rows = []
    if a.workers == 1:
        for j in jobs:
            rows.extend(fit_case(j))
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as pool:
            for result in as_completed([pool.submit(fit_case, j) for j in jobs]):
                result = result.result()
                rows.extend(result)
                print(result[0]["scenario"], result[0]["seed"], flush=True)
    rows.sort(key=lambda r: (r["scenario"], r["seed"], r["method"]))
    csv_write(a.output / "controlled_per_seed.csv", rows)
    (a.output / "manifest.json").write_text(json.dumps({
        "examples_per_scenario": 8192, "train": 5120, "validation": 1024, "test": 2048,
        "target_generator": "Independent staged Picard iteration with fixed splines and known per-input gains",
        "runs": len(rows), "epochs": a.epochs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
