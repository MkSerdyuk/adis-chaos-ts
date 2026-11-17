import numpy as np
import pandas as pd
import pytest

from adis.trading.backtest import run_backtest
from adis.trading.transfer import audit_execution, retire_targets, retired_cash, synchronized_bootstrap


def test_calendar_synchronized_bootstrap_preserves_cross_asset_cancellation():
    first = np.array([-0.2, 0.03, -0.04, 0.01, 0.02] * 20)
    second = first * 0.3
    result = synchronized_bootstrap(np.array([first, second]), np.array([second, first]), repetitions=100)
    assert result["mean_difference"] == result["lower"] == result["upper"] == 0
    assert result["asset_differences"][0] == pytest.approx(-result["asset_differences"][1])
    assert result["asset_interval_confidence"] == 0.975


def test_retirement_exits_after_notice_not_at_last_available_bar():
    index = pd.date_range("2025-05-01", "2025-05-30", freq="h", tz="UTC", inclusive="left")
    close = 10 + np.arange(len(index)) / 1000
    frame = pd.DataFrame(
        dict(Open=close, High=close + 1, Low=close - 1, Close=close, Volume=1.0, Observed=True), index=index
    )
    frame.loc["2025-05-26 03:00":, "Observed"] = False
    notice = dict(
        retirement_decision_bar_utc="2025-05-15 23:00:00+00:00", spot_ceases_utc="2025-05-26 03:00:00+00:00"
    )
    targets = pd.Series(0.95, index=index[index.hour == 23])
    retired = retire_targets(targets, notice)
    assert retired.loc[:"2025-05-14"].eq(0.95).all()
    assert retired.loc["2025-05-15":].eq(0).all()
    result = run_backtest(frame, retired, "2025-05-01", "2025-05-30")
    assert result["trades"].iloc[-1].ExitTime == pd.Timestamp("2025-05-16 01:00", tz="UTC")
    assert audit_execution(frame, result, notice)["retirement_verified"]
    cash = retired_cash(frame, "2025-05-27", "2025-05-30")
    assert cash["metrics"]["total_return"] == cash["metrics"]["es95"] == 0
    assert audit_execution(frame, cash, notice)["trades"] == 0
