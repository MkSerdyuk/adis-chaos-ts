import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TUNED = dict(
    length=3,
    max_gap=3,
    r=5,
    mu=0.02,
    epsilon=0.06502211399080364,
    cluster_eps=0.008371633741976994,
    min_samples=3,
    noise_variance=5.8071705698799575e-05,
    endpoint_eps=0.008371633741976994,
    endpoint_min_samples=3,
    trajectories=16,
    clustered=True,
    patterns=None,
    decay=0.99,
    seed=42,
    noise_schedule="fixed",
    noise_law="normal",
    growth_rule="strict",
)
PAPER = dict(
    length=4,
    max_gap=10,
    r=11,
    mu=0.2,
    epsilon=0.01,
    cluster_eps=0.01,
    min_samples=3,
    noise_variance=1e-5,
    endpoint_eps=0.01,
    endpoint_min_samples=3,
    trajectories=16,
    clustered=True,
    patterns=None,
    decay=0.99,
    seed=42,
    noise_schedule="fixed",
    noise_law="normal",
    growth_rule="strict",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_market(symbol):
    frame = pd.read_csv(DATA / "markets" / f"{symbol}.csv.gz", index_col=0, parse_dates=True)
    if symbol.endswith("USDT"):
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame["Observed"] = frame.Observed.astype(bool)
    return frame


def weekly_decisions():
    sessions = pd.read_csv(DATA / "calendar.csv", index_col=0, parse_dates=True).index
    return pd.Series(sessions, index=sessions).groupby(sessions.to_period("W-FRI")).last().to_numpy()


def start_run(block, config):
    periods = [(config["start"], config["end"])] if block == "btc" else list(config["periods"].values())
    lower, upper = ("2022-01-01", "2026-01-01") if block == "btc" else ("2021-01-01", "2026-11-01")
    for start, end in periods:
        assert lower <= start < end <= upper, (start, end)
    snapshot = json.loads((DATA / "snapshot.json").read_text())
    for name, digest in snapshot["files"].items():
        assert sha256(ROOT / name) == digest, name
    paths = sorted((ROOT / "tools").glob("*.py")) + sorted((ROOT.parent / "src").glob("*.py"))
    paths += [ROOT / f"{block}.py", ROOT.parent / "uv.lock"]
    state = dict(
        block=block,
        config=config,
        source_commit="2050fbfa8f0aa7be573ba21e3af98c88f01ec3a9",
        method_change="Legacy trading predictor replaced by unmodified src.Forecaster; parameters transferred without retuning",
        forecast_source="src.Forecaster.fit/predict; cache keyed by inputs, parameters and implementation",
        comparator_forecast_source="recomputed"
        if config.get("recompute_comparators", False)
        else "original frozen comparator forecasts",
        btc_forest_source="recomputed" if block == "btc" else None,
        snapshot_sha256=sha256(DATA / "snapshot.json"),
        files={str(path.relative_to(ROOT.parent)): sha256(path) for path in paths},
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "numba", "backtesting", "bokeh"]
        },
    )
    identifier = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
    output = ROOT / "results" / block / identifier
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    return output
