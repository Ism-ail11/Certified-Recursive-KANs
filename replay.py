#!/usr/bin/env python3
"""Replay a bundled real-data gate on the matching held-out split."""

import argparse
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from data import DATASETS, SEEDS, load_split
from fit_baselines import load_model
from gate_study import TITLES, anchor_hash, csv_write, load_gate, metric, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("--seed", type=int, choices=SEEDS, default=17)
    p.add_argument("--method", choices=TITLES, default="signed")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baselines", type=Path, default=Path("results/baselines"))
    p.add_argument("--gating", type=Path, default=Path("results/gating"))
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error("Use a positive limit")
    d = load_split(a.dataset, a.seed)
    name = a.method if a.method in ("fixed", "reciprocal", "standard") else "fixed"
    path = a.baselines / a.dataset / name / f"{a.seed}.npz"
    model, _ = load_model(path)
    gate, strength = None, 0.
    if a.method not in ("fixed", "reciprocal", "standard"):
        mode = "signed" if a.method == "signed_full" else a.method
        gate, strength = load_gate(a.gating / a.dataset / str(a.seed) / f"{mode}.npz", anchor_hash(path))
        if a.method == "signed_full":
            strength = 1.
    ids = d.test[:a.limit]
    scores, error, q, fallback, _ = predict(model, d.x[ids], gate, strength)
    _, prediction, _ = metric(d, d.y[ids], scores)
    rows = []
    for i, index in enumerate(ids):
        value = chr(65 + prediction[i]) if d.task == "classification" else float(prediction[i])
        row = dict(test_index=int(index), prediction=value)
        if error is not None:
            row.update(state_error_bound=float(error[i]), certified_at_tolerance=bool(error[i] <= 1e-5),
                       loop_factor=float(q[i]), breakpoint_solves=int(fallback[i]))
        rows.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    csv_write(a.output, rows)
    print(f"Wrote {len(rows)} predictions to {a.output}")


if __name__ == "__main__":
    main()
