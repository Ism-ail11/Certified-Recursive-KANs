"""Reproducible larger-task splits. Features and targets use train-only scaling."""

import csv
import gzip
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


DATASETS = {"bike": "UCI Bike Sharing", "letters": "UCI Letter Recognition"}
DATA_PATH = Path(__file__).resolve().parent / "datasets"
SEEDS = (17, 29, 43, 89, 101)


@dataclass
class Split:
    task: str
    x: np.ndarray
    y: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    groups: np.ndarray
    target_mean: float
    target_scale: float
    n_outputs: int


@lru_cache(maxsize=2)
def raw_data(name):
    if name == "letters":
        with gzip.open(DATA_PATH / "letter_recognition.data.gz", "rt", newline="") as file:
            rows = list(csv.reader(file))
        x = np.asarray([r[1:] for r in rows], dtype=np.float64)
        y = np.asarray([ord(r[0]) - ord("A") for r in rows])
        _, groups = np.unique(x, axis=0, return_inverse=True)
        return x, y, groups
    if name != "bike":
        raise ValueError(f"Unknown dataset {name}")
    with gzip.open(DATA_PATH / "bike_hour.csv.gz", "rt", newline="") as file:
        rows = list(csv.DictReader(file))

    def get(key):
        return np.asarray([float(r[key]) for r in rows])

    time_keys = [(r["dteday"], int(r["hr"])) for r in rows]
    if any(a >= b for a, b in zip(time_keys, time_keys[1:])):
        raise ValueError("Expected strictly chronological hourly rows")
    hour = 2 * np.pi * get("hr") / 24
    weekday = 2 * np.pi * get("weekday") / 7
    month = 2 * np.pi * (get("mnth") - 1) / 12
    weather = np.eye(4)[get("weathersit").astype(int) - 1]
    x = np.column_stack([
        np.sin(hour), np.cos(hour), np.sin(weekday), np.cos(weekday), np.sin(month), np.cos(month),
        get("workingday"), get("holiday"), get("yr"), weather,
        get("temp"), get("atemp"), get("hum"), get("windspeed"),
    ])
    weeks = np.asarray([r["dteday"] for r in rows], dtype="datetime64[D]").astype(np.int64) // 7
    return x, get("cnt"), weeks


@lru_cache(maxsize=16)
def load_split(name, seed):
    x, y, groups = raw_data(name)
    if name == "bike":
        train, valid, test = np.split(np.arange(len(y)), [int(0.64 * len(y)), int(0.80 * len(y))])
        task = "regression"
        target_mean, target_scale = float(y[train].mean()), float(y[train].std())
        outcomes = (y - target_mean) / target_scale
        n_outputs = 1
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fit, test = next(splitter.split(x, y, groups))
        holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1000)
        a, b = next(holdout.split(x[fit], y[fit], groups[fit]))
        train, valid = fit[a], fit[b]
        task, outcomes, n_outputs = "classification", y, 26
        target_mean, target_scale = 0.0, 1.0
    center = x[train].mean(axis=0)
    spread = x[train].std(axis=0)
    return Split(task, (x - center) / np.where(spread < 1e-12, 1.0, spread), outcomes,
                 train, valid, test, groups, target_mean, target_scale, n_outputs)
