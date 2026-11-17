import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ROOT = Path("results/trading/noncrypto")
DATA = Path("data/trading/noncrypto")
PROTOCOL = Path("research/trading/noncrypto/protocol.json")


def calendar():
    return xcals.get_calendar("XNYS", start="2004-12-01", end="2026-11-30")


def weekly_decisions():
    sessions = calendar().sessions
    return pd.Series(sessions, index=sessions).groupby(sessions.to_period("W-FRI")).last().to_numpy()


def parse_chart(payload):
    assert payload["chart"]["error"] is None
    result = payload["chart"]["result"][0]
    assert result["meta"]["currency"] == "USD"
    dates = (
        pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York")
        .tz_localize(None)
        .normalize()
    )
    frame = pd.DataFrame(result["indicators"]["quote"][0], index=dates).rename(columns=str.title)
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert np.isfinite(frame.to_numpy()).all() and (frame > 0).all().all()
    assert (frame.High >= frame[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (frame.Low <= frame[["Open", "Close", "High"]].min(axis=1)).all()
    frame["Dividend"] = 0.0
    frame["Split"] = 1.0
    events = result.get("events", {})
    for event in events.get("dividends", {}).values():
        date = (
            pd.Timestamp(event["date"], unit="s", tz="UTC")
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )
        if date < frame.index[0]:
            continue
        assert date in frame.index
        frame.loc[date, "Dividend"] += event["amount"]
    for event in events.get("splits", {}).values():
        date = (
            pd.Timestamp(event["date"], unit="s", tz="UTC")
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )
        if date < frame.index[0]:
            continue
        assert date in frame.index
        frame.loc[date, "Split"] *= event["numerator"] / event["denominator"]
    assert not events.get("capitalGains"), "Additional capital-gain distributions require explicit accounting"
    frame["FutureSplitFactor"] = frame.Split.iloc[::-1].cumprod().iloc[::-1] / frame.Split
    for column in ["Open", "High", "Low", "Close", "Dividend"]:
        frame["Raw" + column] = frame[column] * frame.FutureSplitFactor
    frame.index.name = "session"
    return frame


def load_asset(symbol):
    return pd.read_csv(DATA / f"{symbol}.csv.gz", index_col=0, parse_dates=True)


def download():
    protocol = json.loads(PROTOCOL.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    query = urlencode(
        dict(
            period1=int(pd.Timestamp(protocol["snapshot_start"], tz="UTC").timestamp()),
            period2=int(pd.Timestamp(protocol["snapshot_end_exclusive"], tz="UTC").timestamp()),
            interval="1d",
            events="div,splits,capitalGains",
            includeAdjustedClose="true",
        )
    )
    manifests = []
    for symbol in protocol["instruments"]:
        path = DATA / f"{symbol}.json"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
        if not path.exists():
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            path.write_bytes(urlopen(request, timeout=60).read())
        payload = json.loads(path.read_text())
        frame = parse_chart(payload)
        expected = calendar().sessions_in_range(
            protocol["snapshot_start"],
            pd.Timestamp(protocol["snapshot_end_exclusive"]) - pd.Timedelta(days=1),
        )
        assert frame.index.equals(expected), dict(
            symbol=symbol,
            missing=expected.difference(frame.index).astype(str).tolist(),
            extra=frame.index.difference(expected).astype(str).tolist(),
        )
        output = DATA / f"{symbol}.csv.gz"
        frame.to_csv(output, compression={"method": "gzip", "mtime": 0})
        manifest = dict(
            symbol=symbol,
            source=url,
            raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            sessions=len(frame),
            first=str(frame.index[0]),
            last=str(frame.index[-1]),
            splits=frame.loc[frame.Split != 1, "Split"]
            .rename_axis("date")
            .reset_index()
            .assign(date=lambda x: x.date.astype(str))
            .to_dict("records"),
            dividend_events=int((frame.Dividend > 0).sum()),
        )
        manifests.append(manifest)
        print(
            symbol,
            "sessions",
            len(frame),
            "dividends",
            manifest["dividend_events"],
            "splits",
            len(manifest["splits"]),
            flush=True,
        )
    (ROOT / "data_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    schedule = calendar().schedule
    schedule.to_csv(ROOT / "calendar.csv")
    reference = json.loads(Path("research/trading/noncrypto/corporate_action_reference.json").read_text())
    checks = []
    for row in reference["checks"]:
        actual = float(load_asset(row["symbol"]).loc[row["date"], row["column"]])
        expected = round(row["issuer_value"], row["provider_decimals"])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        checks.append(dict(**row, actual=actual, difference_from_issuer=actual - row["issuer_value"]))
    (ROOT / "corporate_action_checks.json").write_text(
        json.dumps(dict(checks=checks, limitation=reference["limitation"]), indent=2) + "\n"
    )


if __name__ == "__main__":
    download()
