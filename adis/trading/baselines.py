import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def technical_targets(frame):
    daily = frame.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    close = daily.Close
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = -change.clip(upper=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / down)
    rsi_state = pd.Series(np.where(rsi < 30, 1.0, np.where(rsi > 50, 0.0, np.nan)), index=daily.index)
    high = daily.High.rolling(20).max().shift(1)
    low = daily.Low.rolling(10).min().shift(1)
    breakout = pd.Series(np.where(close > high, 1.0, np.where(close < low, 0.0, np.nan)), index=daily.index)
    volatility = np.log(close).diff().rolling(30).std() * np.sqrt(365)
    targets = pd.DataFrame(
        {
            "cash": 0.0,
            "buy_hold": 0.95,
            "sma_50_200_daily": 0.95 * (close.rolling(50).mean() > close.rolling(200).mean()),
            "momentum_90_daily": 0.95 * (close.pct_change(90) > 0),
            "donchian_20_10_daily": 0.95 * breakout.ffill().fillna(0),
            "rsi_14_daily": 0.95 * rsi_state.ffill().fillna(0),
            "vol_target_20pct_30daily": (0.20 / volatility).clip(lower=0, upper=0.95).fillna(0),
        },
        index=daily.index,
    )
    targets.index += pd.Timedelta(hours=23)
    return targets


def forest_targets(frame, start, end):
    close = frame.Close
    returns = np.log(close).diff()
    features = pd.DataFrame(
        {f"momentum_{lag}": np.log(close / close.shift(lag)) for lag in [1, 3, 6, 12, 24, 72, 168]}
    )
    for lag in [24, 72, 168]:
        features[f"volatility_{lag}"] = returns.rolling(lag).std()
    features["volume_ratio"] = frame.Volume / frame.Volume.rolling(168).mean()
    labels = close.shift(-24) / close - 1
    outputs, fits = [], []
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    for quarter in pd.date_range(start, end, freq="QS", inclusive="left"):
        boundary = frame.index.searchsorted(quarter)
        indices = np.arange(boundary - 10000, boundary - 24)
        assert frame.index[indices[-1] + 24] < quarter
        x, y = features.iloc[indices], labels.iloc[indices]
        assert np.isfinite(x).all().all() and np.isfinite(y).all()
        model = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50, max_features=0.8, random_state=42, n_jobs=4
        )
        model.fit(x, y)
        dates = frame.index[
            (frame.index >= quarter)
            & (frame.index < min(quarter + pd.DateOffset(months=3), end))
            & (frame.index.hour == 23)
        ]
        prediction = model.predict(features.loc[dates])
        outputs.append(pd.Series(np.where(prediction > 0.003, 0.95, 0), index=dates))
        fits.append(
            dict(
                quarter=str(quarter),
                fit_start=str(x.index[0]),
                last_feature=str(x.index[-1]),
                last_label=str(frame.index[indices[-1] + 24]),
            )
        )
    return pd.concat(outputs), fits
