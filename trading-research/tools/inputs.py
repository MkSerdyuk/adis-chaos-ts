import json

import pandas as pd

from .data import DATA, ROOT, sha256
from .forecasts import walk_forward
from .predictors import forecast_models


def hourly_forecast(frame, symbol, name, parameters, output, start, end, seed=None):
    settings = parameters if seed is None else dict(parameters, seed=seed)
    ledger = walk_forward(frame, start, end, settings)
    folder = output / "forecasts" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(folder / f"{name}.csv")
    return ledger


def daily_forecast(frame, symbol, name, parameters, output, seed=None):
    settings = parameters if seed is None else dict(parameters, seed=seed)
    ledger = walk_forward(frame, "2021-01-01", "2026-11-01", settings, training=2520, horizon=5, hourly=False)
    folder = output / "forecasts" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(folder / f"{name}.csv")
    return ledger


def predictors(frame, symbol, decisions, recompute):
    if recompute:
        return forecast_models(frame, decisions)
    frozen = json.loads((DATA / "snapshot.json").read_text())["comparator_code"]
    assert sha256(ROOT / "tools/predictors.py") == frozen, (
        "Changed comparators require recompute_comparators=True"
    )
    return pd.read_csv(DATA / "forecasts" / symbol / "predictors.csv", parse_dates=["decision_session"])
