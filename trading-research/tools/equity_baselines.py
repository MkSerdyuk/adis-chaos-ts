import numpy as np
import pandas as pd


def technical_targets(frame, decisions):
    close = frame.Close
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = -change.clip(upper=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / down)
    rsi_state = pd.Series(np.where(rsi < 30, 1.0, np.where(rsi > 50, 0.0, np.nan)), index=frame.index)
    breakout = pd.Series(
        np.where(
            close > frame.High.rolling(20).max().shift(1),
            1.0,
            np.where(close < frame.Low.rolling(10).min().shift(1), 0.0, np.nan),
        ),
        index=frame.index,
    )
    volatility = np.log(close).diff().rolling(30).std() * np.sqrt(252)
    targets = pd.DataFrame(
        dict(
            cash=0.0,
            buy_hold=0.95,
            sma50_200=0.95 * (close.rolling(50).mean() > close.rolling(200).mean()),
            momentum90=0.95 * (close.pct_change(90) > 0),
            donchian20_10=0.95 * breakout.ffill().fillna(0),
            rsi14=0.95 * rsi_state.ffill().fillna(0),
            vol_target20pct_30=(0.2 / volatility).clip(0, 0.95).fillna(0),
        ),
        index=frame.index,
    )
    return targets.reindex(decisions)
