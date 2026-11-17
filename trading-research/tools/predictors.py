import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def forecast_models(frame, decisions):
    close = frame.Close
    returns = np.log(close).diff()
    features = pd.DataFrame({f"momentum_{lag}": np.log(close / close.shift(lag)) for lag in [1, 5, 20, 60]})
    for lag in [20, 60]:
        features[f"volatility_{lag}"] = returns.rolling(lag).std()
    features["volume_ratio"] = frame.Volume / frame.Volume.rolling(20).mean()
    labels = close.shift(-5) / close - 1
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50, max_features=0.8, random_state=42, n_jobs=4
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "svr": make_pipeline(StandardScaler(), SVR(C=1, epsilon=0.01, gamma="scale")),
    }
    chunks = []
    for quarter in pd.date_range("2021-01-01", "2026-10-01", freq="QS"):
        boundary = frame.index.searchsorted(quarter)
        rows = np.arange(boundary - 2520, boundary - 5)
        assert frame.index[rows[-1] + 5] < quarter
        dates = decisions[(decisions >= quarter) & (decisions < quarter + pd.DateOffset(months=3))]
        assert np.isfinite(features.iloc[rows]).all().all() and np.isfinite(labels.iloc[rows]).all()
        for name, specification in models.items():
            model = clone(specification).fit(features.iloc[rows], labels.iloc[rows])
            prediction = model.predict(features.loc[dates])
            chunks.append(
                pd.DataFrame(
                    dict(
                        decision_session=dates,
                        model=name,
                        predicted_return=prediction,
                        target=np.where(prediction > 0.003, 0.95, 0),
                        fit_start=str(frame.index[rows[0]]),
                        last_label_session=str(frame.index[rows[-1] + 5]),
                        quarter=str(quarter),
                    )
                )
            )
    result = pd.concat(chunks, ignore_index=True)
    return result
