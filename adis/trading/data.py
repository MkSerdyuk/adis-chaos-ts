import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


def download_month(symbol, month, directory):
    name = f"{symbol}-1h-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{name}"
    archive, checksum = directory / name, directory / f"{name}.CHECKSUM"
    if not checksum.exists():
        checksum.write_bytes(urlopen(url + ".CHECKSUM", timeout=60).read())
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        temporary.write_bytes(urlopen(url, timeout=60).read())
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == checksum.read_text().split()[0], name
    with ZipFile(archive) as zipped:
        assert len(zipped.namelist()) == 1
        raw = pd.read_csv(io.BytesIO(zipped.read(zipped.namelist()[0])), header=None)
    unit = "us" if month >= "2025-01" else "ms"
    zero_volume = raw.loc[raw[5] == 0, 0].tolist()
    raw = raw.loc[raw[5] > 0].reset_index(drop=True)
    index = pd.to_datetime(raw[0], unit=unit, utc=True)
    close_time = pd.to_datetime(raw[6], unit=unit, utc=True)
    assert (close_time >= index).all()
    assert (close_time < index + pd.Timedelta(hours=1)).all()
    frame = raw.iloc[:, 1:6].astype(float)
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    frame.index = pd.DatetimeIndex(index, name="time")
    assert (frame.index.strftime("%Y-%m") == month).all()
    return frame, dict(
        url=url, sha256=digest, rows=len(frame), timestamp_unit=unit, zero_volume_open_times=zero_volume
    )


def load_market(directory=Path("data/trading")):
    frame = pd.read_csv(directory / "BTCUSDT_1h.csv.gz", index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame["Observed"] = frame["Observed"].astype(bool)
    return frame


def download_market(start="2019-01", end="2025-12", directory=Path("data/trading"), workers=4):
    directory.mkdir(parents=True, exist_ok=True)
    archives = directory / "archives"
    archives.mkdir(exist_ok=True)
    months = pd.period_range(start, end, freq="M").astype(str)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        downloaded = list(pool.map(lambda month: download_month("BTCUSDT", month, archives), months))
    frame = pd.concat([item[0] for item in downloaded]).sort_index()
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert (frame.index.minute == 0).all() and (frame.index.second == 0).all()
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (frame.High >= frame[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (frame.Low <= frame[["Open", "Close", "High"]].min(axis=1)).all()
    assert (frame.Volume >= 0).all()
    expected = pd.date_range(f"{start}-01", pd.Period(end).end_time.floor("h"), freq="h", tz="UTC")
    missing = expected.difference(frame.index)
    assert len(missing) / len(expected) < 0.005
    frame["Observed"] = True
    frame = frame.reindex(expected)
    frame["Observed"] = frame.Observed.eq(True)
    previous = frame.Close.ffill()
    for column in ["Open", "High", "Low", "Close"]:
        frame.loc[~frame.Observed, column] = previous[~frame.Observed]
    frame.loc[~frame.Observed, "Volume"] = 0.0
    assert np.isfinite(frame[["Open", "High", "Low", "Close", "Volume"]]).all().all()
    frame.index.name = "time"
    frame.to_csv(directory / "BTCUSDT_1h.csv.gz", compression={"method": "gzip", "mtime": 0})
    manifest = dict(
        symbol="BTCUSDT",
        interval="1h",
        start=str(frame.index[0]),
        end=str(frame.index[-1]),
        rows=len(frame),
        missing_hours=len(missing),
        missing_times=missing.astype(str).tolist(),
        archives=[item[1] for item in downloaded],
    )
    manifest["csv_sha256"] = hashlib.sha256((directory / "BTCUSDT_1h.csv.gz").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"archives", "missing_times"}}, indent=2))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("data/trading"))
    args = parser.parse_args()
    download_market(directory=args.directory)
