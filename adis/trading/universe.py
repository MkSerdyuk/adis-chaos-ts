import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd

from .data import download_month

ROOT = Path("results/trading/transfer")
PROTOCOL = Path("research/trading/transfer_protocol.json")
DATA = Path("data/trading/transfer")


def archived_symbols():
    symbols, marker = [], ""
    namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        query = urlencode(
            dict(delimiter="/", prefix="data/spot/monthly/klines/", marker=marker, **{"max-keys": 1000})
        )
        raw = urlopen(
            "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?" + query, timeout=60
        ).read()
        root = ElementTree.fromstring(raw)
        symbols.extend(
            node.text.split("/")[-2] for node in root.findall("s:CommonPrefixes/s:Prefix", namespace)
        )
        if root.findtext("s:IsTruncated", namespaces=namespace) == "false":
            return sorted(symbol for symbol in symbols if symbol.endswith("USDT"))
        marker = root.findtext("s:NextMarker", namespaces=namespace)
        assert marker


def historical_candidate(symbol):
    directory = DATA / symbol / "archives"
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-2019-01.zip.CHECKSUM"
    checksum = directory / f"{symbol}-1h-2019-01.zip.CHECKSUM"
    if not checksum.exists():
        try:
            checksum.write_bytes(urlopen(quote(url, safe=":/"), timeout=60).read())
        except HTTPError as error:
            if error.code != 404:
                raise
            return dict(symbol=symbol, eligible=False, reason="no January 2019 archive")
    frames, archives, quote_volume = [], [], 0.0
    for month in pd.period_range("2019-01", "2019-12", freq="M").astype(str):
        try:
            frame, metadata = download_month(symbol, month, directory)
        except HTTPError as error:
            if error.code != 404:
                raise
            return dict(symbol=symbol, eligible=False, reason=f"missing {month} archive")
        frames.append(frame)
        archives.append(metadata)
        with ZipFile(directory / f"{symbol}-1h-{month}.zip") as zipped:
            raw = pd.read_csv(io.BytesIO(zipped.read(zipped.namelist()[0])), header=None, usecols=[7])
            quote_volume += float(raw[7].sum())
    frame = pd.concat(frames)
    assert frame.index.is_unique
    eligible = frame.index[0] == pd.Timestamp("2019-01-01", tz="UTC") and len(frame) >= 8760 * 0.99
    print(symbol, len(frame), round(quote_volume), eligible, flush=True)
    return dict(
        symbol=symbol,
        eligible=eligible,
        reason="eligible" if eligible else "late start or less than 99% observed hours in 2019",
        first=str(frame.index[0]),
        observed_hours=len(frame),
        quote_volume_2019=quote_volume,
        archives=archives,
    )


def select_universe():
    protocol = json.loads(PROTOCOL.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "universe.json"
    assert not path.exists(), "Universe already frozen; inspect the saved selection rather than overwrite it"
    symbols = archived_symbols()
    exclusions, candidates = [], []
    for symbol in symbols:
        base = symbol[:-4]
        if base in protocol["excluded_base_assets"] or any(
            base.endswith(suffix) for suffix in protocol["excluded_token_suffixes"]
        ):
            exclusions.append(dict(symbol=symbol, eligible=False, reason="prespecified asset category"))
        else:
            candidates.append(symbol)
    print(f"Checking {len(candidates)} archived USDT pairs", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(historical_candidate, candidates))
    ranked = sorted(
        (row for row in rows if row["eligible"]), key=lambda row: (-row["quote_volume_2019"], row["symbol"])
    )
    selected = [row["symbol"] for row in ranked[: protocol["universe_count"]]]
    assert len(selected) == protocol["universe_count"]
    result = dict(
        selected=selected,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        universe_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ranked=[{key: value for key, value in row.items() if key != "archives"} for row in ranked],
        candidates=rows + exclusions,
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Frozen universe:", selected, flush=True)


if __name__ == "__main__":
    select_universe()
