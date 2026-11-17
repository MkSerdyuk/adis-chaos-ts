#!/usr/bin/env bash
set -eu

uv sync --locked --extra trading
uv run python - <<'PY'
import gzip
import hashlib
import json
from pathlib import Path

root = Path("results/trading/noncrypto")
destination = Path("data/trading/noncrypto")
destination.mkdir(parents=True, exist_ok=True)
identity = json.loads(Path("results/trading/selected.json").read_text())["identity"]
for source_name, target_name, key in [
    ("BTCUSDT_1h.csv.gz", "BTCUSDT_1h.csv.gz", "data_sha256"),
    ("btc_manifest.json", "manifest.json", "manifest_sha256"),
]:
    content = (root / "snapshots" / source_name).read_bytes()
    assert hashlib.sha256(content).hexdigest() == identity[key]
    path = Path("data/trading") / target_name
    if path.exists():
        assert path.read_bytes() == content
    else:
        path.write_bytes(content)
for item in json.loads((root / "data_manifest.json").read_text()):
    source = root / "snapshots" / f"{item['symbol']}.json.gz"
    content = gzip.decompress(source.read_bytes())
    assert hashlib.sha256(content).hexdigest() == item["raw_sha256"]
    path = destination / f"{item['symbol']}.json"
    if path.exists():
        assert path.read_bytes() == content
    else:
        path.write_bytes(content)
PY
uv run --extra trading python -m adis.trading.equity_data
uv run --extra trading python -m adis.trading.equity_experiment
uv run --extra trading python -m adis.trading.equity_audit
uv run --extra trading python -m adis.trading.equity_report
uv run --extra trading python -m adis.trading.predictive_comparison
uv run --extra trading python -m adis.trading.predictive_audit
uv run --extra trading python -m adis.trading.predictive_report
