import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from numba import njit

OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
OPSD_COLUMN = "DE_load_actual_entsoe_transparency"


@njit(cache=True)
def _lorenz_rhs(state, sigma, beta, rho):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


@njit(cache=True)
def lorenz(n, discard=3000, dt=0.1, initial=(1.0, 1.0, 1.0), sigma=10.0, beta=8 / 3, rho=28.0, component=0):
    state = np.array(initial)
    values = np.empty(n)
    for i in range(n + discard):
        if i >= discard:
            values[i - discard] = state[component]
        a = _lorenz_rhs(state, sigma, beta, rho)
        b = _lorenz_rhs(state + dt * a / 2, sigma, beta, rho)
        c = _lorenz_rhs(state + dt * b / 2, sigma, beta, rho)
        d = _lorenz_rhs(state + dt * c, sigma, beta, rho)
        state += dt * (a + 2 * b + 2 * c + d) / 6
    return values


def german_load(directory="data"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "opsd_2020.csv"
    if not raw.exists():
        urlretrieve(OPSD_URL, raw)
    frame = pd.read_csv(raw, usecols=["utc_timestamp", OPSD_COLUMN])
    frame["utc_timestamp"] = pd.to_datetime(frame["utc_timestamp"], utc=True)
    frame = frame.set_index("utc_timestamp").loc["2015-01-01":"2016-02-20"]
    if frame[OPSD_COLUMN].isna().any() or not frame.index.equals(
        pd.date_range(frame.index[0], frame.index[-1], freq="h")
    ):
        raise ValueError("German load must be complete and hourly; no imputation is applied")
    frame.rename(columns={OPSD_COLUMN: "load_mw"}).to_csv(directory / "germany.csv")
    manifest = {
        "url": OPSD_URL,
        "version": "2020-10-06",
        "column": OPSD_COLUMN,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
        "observations": len(frame),
        "timezone": "UTC",
        "missing": 0,
        "boundary": "2014-12-31T23:00Z is missing in source; start fixed at 2015-01-01T00:00Z",
    }
    (directory / "germany_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frame[OPSD_COLUMN].to_numpy(), frame.index


def windows(series, origins, context, horizon):
    origins = np.asarray(origins)
    histories = series[origins[:, None] - np.arange(context - 1, -1, -1)]
    targets = series[origins[:, None] + np.arange(1, horizon + 1)]
    return histories, targets
