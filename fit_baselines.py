#!/usr/bin/env python3
"""Refit the cubic, fixed paired, and original reciprocal paired baselines."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from data import DATASETS, SEEDS, load_split
from paired_kan import CubicModel
from splines import AdamW, loss_gradient


METHODS = ("fixed", "reciprocal", "standard")


def save_model(path, model, settings):
    data = {f"p{i:03}": p for i, p in enumerate(model.snapshot())}
    data["metadata"] = np.array(json.dumps(settings, allow_nan=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        model = CubicModel(metadata["inputs"], metadata["outputs"], metadata["seed"],
                           metadata["method"], metadata["reference_width"])
        model.restore([stored[k] for k in sorted(stored.files) if k.startswith("p")])
    return model, metadata


def loss_on(model, x, y, task, batch=256):
    scores = np.concatenate([model.from_features(model.encode(x[k:k + batch]))[0]
                             for k in range(0, len(x), batch)])
    return loss_gradient(scores, y, task)[0]


def train(job):
    root, dataset, method, seed, config = job
    d = load_split(dataset, seed)
    path = root / dataset / method / f"{seed}.npz"
    trace = path.with_suffix(".json")
    expected = dict(dataset=dataset, method=method, seed=seed, inputs=d.x.shape[1], outputs=d.n_outputs,
                    reference_width=24, configuration=config)
    if path.exists() and trace.exists():
        with np.load(path, allow_pickle=False) as stored:
            existing = json.loads(str(stored["metadata"]))
        if all(existing[k] == v for k, v in expected.items()):
            return existing
    model = CubicModel(d.x.shape[1], d.n_outputs, seed, method)
    params = model.parameters()
    opt = AdamW(params)
    rand = np.random.default_rng(seed + 91007)
    best, history = np.inf, []
    before = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        indexes = rand.permutation(d.train)
        lr = config["lr"] * (.25 + .75 * (1 + np.cos(np.pi * (epoch - 1) / config["epochs"])) / 2)
        train_loss = 0.
        for offset in range(0, len(indexes), config["batch"]):
            rows = indexes[offset:offset + config["batch"]]
            for _, g in params:
                g.fill(0.)
            score, cache = model.training_forward(d.x[rows])
            loss, grad = loss_gradient(score, d.y[rows], d.task)
            model.backward(grad, cache)
            opt.update(lr)
            model.project()
            train_loss += len(rows) * loss
        val = loss_on(model, d.x[d.valid], d.y[d.valid], d.task)
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation score")
        history.append({"epoch": epoch, "train_loss": train_loss / len(indexes), "validation_loss": val})
        if val < best - 1e-7:
            best, best_epoch, weights = val, epoch, model.snapshot()
        if epoch - best_epoch >= config["patience"]:
            break
    model.restore(weights)
    result = dict(expected, validation_loss=best, best_epoch=best_epoch, epochs_run=len(history),
                  params=model.parameter_count, train_seconds=perf_counter() - before)
    save_model(path, model, result)
    trace.write_text(json.dumps(dict(result, history=history), indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/baselines"))
    p.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    p.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=48)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=.006)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if min(a.epochs, a.batch, a.patience, a.workers) < 1 or a.lr <= 0:
        p.error("Use positive training settings")
    cfg = dict(epochs=a.epochs, batch=a.batch, patience=a.patience, lr=a.lr)
    jobs = [(a.output, d, m, s, cfg) for d in a.datasets for s in a.seeds for m in a.methods]

    def show(i, r):
        print(f"[{i:2d}/{len(jobs)}] {r['dataset']:7} {r['method']:11} seed={r['seed']:3d}"
              f" epoch={r['best_epoch']:2d} val={r['validation_loss']:.6f}", flush=True)

    if a.workers == 1:
        for i, job in enumerate(jobs, 1):
            show(i, train(job))
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as pool:
            for i, result in enumerate(as_completed([pool.submit(train, j) for j in jobs]), 1):
                show(i, result.result())
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "manifest.json").write_text(json.dumps({
        "datasets": a.datasets, "methods": list(a.methods), "seeds": list(a.seeds), "runs": len(jobs), **cfg,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
