import hashlib
import importlib.metadata
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data():
    protocol = json.loads((DATA / "protocol.json").read_text())
    sources = json.loads((DATA / "raw/sources.json").read_text())
    for source in sources:
        assert sha256(DATA / "raw" / source["file"]) == source["sha256"]
    summaries = []
    for station, name in protocol["stations"].items():
        pieces = []
        for pattern in [f"*{station}_akt.zip", f"*{station}_*_hist.zip"]:
            (path,) = (DATA / "raw").glob(pattern)
            with ZipFile(path) as archive:
                (product,) = [name for name in archive.namelist() if name.startswith("produkt_")]
                frame = pd.read_csv(archive.open(product), sep=";")
            frame.columns = frame.columns.str.strip()
            assert (frame.STATIONS_ID == int(station)).all()
            pieces.append(frame)
        raw = pd.concat(pieces).drop_duplicates("MESS_DATUM", keep="last").sort_values("MESS_DATUM")
        index = pd.to_datetime(raw.MESS_DATUM.astype(str), format="%Y%m%d%H", utc=True)
        frame = pd.DataFrame(
            dict(Temperature=raw.TT_TU.replace(-999, np.nan).to_numpy(), Quality=raw.QN_9.to_numpy()),
            index=index,
        )
        index = pd.date_range(
            protocol["snapshot_start"],
            protocol["snapshot_end_exclusive"],
            inclusive="left",
            freq="h",
            tz="UTC",
        )
        frame = frame.reindex(index)
        frame["Observed"] = frame.Temperature.notna()
        frame["ModelTemperature"] = frame.Temperature.ffill()
        assert frame.ModelTemperature.notna().all()
        frame.index.name = "time"
        frame.to_csv(DATA / "stations" / f"{station}.csv.gz", compression=dict(method="gzip", mtime=0))
        summaries.append(
            dict(
                station=station,
                name=name,
                hours=len(frame),
                observed=int(frame.Observed.sum()),
                missing=int((~frame.Observed).sum()),
            )
        )
    pd.DataFrame(summaries).to_csv(DATA / "coverage.csv", index=False)
    files = {
        str(path.relative_to(DATA)): sha256(path)
        for path in sorted(DATA.rglob("*"))
        if path.is_file() and path.name != "snapshot.json"
    }
    (DATA / "snapshot.json").write_text(json.dumps(dict(files=files), indent=2) + "\n")


def load_station(station):
    frame = pd.read_csv(DATA / "stations" / f"{station}.csv.gz", index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame["Observed"] = frame.Observed.astype(bool)
    frame["Eligible"] = frame.Observed.rolling(169, min_periods=169).sum().eq(169)
    return frame


def start_run(config):
    snapshot = json.loads((DATA / "snapshot.json").read_text())
    for name, digest in snapshot["files"].items():
        assert sha256(DATA / name) == digest, name
    paths = sorted((ROOT / "tools").glob("*.py")) + sorted((ROOT.parent / "src").glob("*.py"))
    paths += [ROOT / "weather.py", ROOT.parent / "uv.lock"]
    state = dict(
        config=config,
        snapshot_sha256=sha256(DATA / "snapshot.json"),
        files={str(path.relative_to(ROOT.parent)): sha256(path) for path in paths},
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "numba", "scikit-learn", "bokeh", "joblib"]
        },
    )
    identifier = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:20]
    output = ROOT / "results" / identifier
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    return output
