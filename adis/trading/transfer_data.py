import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_month
from .universe import DATA, ROOT

RETIREMENT = Path("research/trading/transfer_retirement.json")


def retirement(symbol):
    return json.loads(RETIREMENT.read_text()).get(symbol)


def load_asset(symbol):
    frame = pd.read_csv(DATA / symbol / "market.csv.gz", index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    assert frame.Observed.dtype == bool
    return frame


def download_asset(symbol):
    directory = DATA / symbol
    archives = directory / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    notice = retirement(symbol)
    cessation = pd.Timestamp(notice["spot_ceases_utc"]) if notice else pd.Timestamp("2026-11-01", tz="UTC")
    months = pd.period_range(
        "2019-01", (cessation - pd.Timedelta(hours=1)).strftime("%Y-%m"), freq="M"
    ).astype(str)
    with ThreadPoolExecutor(max_workers=8) as pool:
        downloaded = list(pool.map(lambda month: download_month(symbol, month, archives), months))
    frame = pd.concat([item[0] for item in downloaded]).sort_index()
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    expected = pd.date_range("2019-01-01", "2026-11-01", freq="h", tz="UTC", inclusive="left")
    assert frame.index.isin(expected).all()
    assert (frame.index < cessation).all()
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (frame.High >= frame[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (frame.Low <= frame[["Open", "Close", "High"]].min(axis=1)).all()
    active_grid = expected[expected < cessation]
    missing = active_grid.difference(frame.index)
    assert len(missing) / len(active_grid) < 0.005
    frame["Observed"] = True
    frame = frame.reindex(expected)
    frame["Observed"] = frame.Observed.eq(True)
    previous = frame.Close.ffill()
    for column in ["Open", "High", "Low", "Close"]:
        frame.loc[~frame.Observed, column] = previous[~frame.Observed]
    frame.loc[~frame.Observed, "Volume"] = 0.0
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    frame.index.name = "time"
    path = directory / "market.csv.gz"
    frame.to_csv(path, compression={"method": "gzip", "mtime": 0})
    manifest = dict(
        symbol=symbol,
        interval="1h",
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        missing_active_hours=len(missing),
        missing_times=missing.astype(str).tolist(),
        last_observed=str(frame.index[frame.Observed][-1]),
        retirement=notice,
        archives=[item[1] for item in downloaded],
    )
    (ROOT / symbol).mkdir(parents=True, exist_ok=True)
    (ROOT / symbol / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(symbol, "downloaded", "missing active hours:", len(missing), flush=True)


if __name__ == "__main__":
    for symbol in json.loads((ROOT / "universe.json").read_text())["selected"]:
        download_asset(symbol)
