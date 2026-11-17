import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import ROOT, sha256
from .forecasts import core_identity, fingerprint

NAMES = [
    "persistence",
    "daily_seasonal",
    "harmonic_climatology",
    "ridge",
    "svr",
    "random_forest",
    "gradient_boosting",
]


def calendar_features(index):
    year = 2 * np.pi * (index.dayofyear.to_numpy() - 1 + index.hour.to_numpy() / 24) / 365.25
    day = 2 * np.pi * index.hour.to_numpy() / 24
    return np.column_stack(
        [
            np.ones(len(index)),
            np.sin(year),
            np.cos(year),
            np.sin(2 * year),
            np.cos(2 * year),
            np.sin(day),
            np.cos(day),
            np.sin(2 * day),
            np.cos(2 * day),
            np.sin(year) * np.sin(day),
            np.cos(year) * np.cos(day),
        ]
    )


def features(frame):
    temperature = frame.ModelTemperature
    values = pd.DataFrame(
        {f"lag_{lag}": temperature.shift(lag) for lag in [0, 1, 2, 3, 6, 12, 24, 48, 72, 168]}
    )
    for window in [24, 168]:
        values[f"mean_{window}"] = temperature.rolling(window).mean()
        values[f"std_{window}"] = temperature.rolling(window).std()
    calendar = calendar_features(frame.index)
    for i in range(1, calendar.shape[1]):
        values[f"calendar_{i}"] = calendar[:, i]
    return values


def predict(frame, quarter, origins, training, horizons):
    boundary = frame.index.searchsorted(quarter)
    state = dict(
        quarter=str(quarter),
        origins=origins.astype(str).tolist(),
        training=training,
        horizons=horizons,
        data_sha256=hashlib.sha256(
            pd.util.hash_pandas_object(
                frame.loc[frame.index <= origins[-1], ["ModelTemperature", "Observed", "Eligible"]],
                index=True,
            )
            .to_numpy()
            .tobytes()
        ).hexdigest(),
        code=sha256(ROOT / "tools/baselines.py"),
        versions=core_identity()["versions"],
    )
    directory = ROOT / "results/cache/baselines"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fingerprint(state)}.npz"
    if path.exists():
        assert json.loads(path.with_suffix(".json").read_text()) == state
        data = np.load(path)
        return {name: data[name] for name in NAMES}, json.loads(path.with_suffix(".fit.json").read_text())
    x = features(frame)
    labels = pd.DataFrame({h: frame.Temperature.shift(-h) - frame.ModelTemperature for h in horizons})
    rows = np.arange(boundary - training, boundary - max(horizons))
    assert rows[0] >= 0 and frame.index[rows[-1] + max(horizons)] < quarter
    valid = (
        frame.Eligible.iloc[rows].to_numpy()
        & x.iloc[rows].notna().all(axis=1).to_numpy()
        & labels.iloc[rows].notna().all(axis=1).to_numpy()
    )
    rows = rows[valid]
    assert len(rows) > 1000
    eligible = frame.Eligible.reindex(origins).to_numpy()
    dates = origins[eligible]
    models = dict(
        ridge=make_pipeline(StandardScaler(), Ridge(alpha=10)),
        svr=MultiOutputRegressor(
            make_pipeline(
                StandardScaler(),
                TransformedTargetRegressor(
                    regressor=SVR(C=1, epsilon=0.01, gamma="scale"), transformer=StandardScaler()
                ),
            )
        ),
        random_forest=RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50, max_features=0.8, random_state=42, n_jobs=4
        ),
        gradient_boosting=MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1,
                early_stopping=False,
                random_state=42,
            )
        ),
    )
    predictions = {name: np.full((len(origins), len(horizons)), np.nan) for name in NAMES}
    for name, specification in models.items():
        model = clone(specification).fit(x.iloc[rows], labels.iloc[rows])
        if len(dates):
            predictions[name][eligible] = (
                model.predict(x.loc[dates]) + frame.ModelTemperature.reindex(dates).to_numpy()[:, None]
            )
    predictions["persistence"][eligible] = frame.ModelTemperature.reindex(dates).to_numpy()[:, None]
    train = frame.iloc[boundary - training : boundary]
    observed = train.Observed.to_numpy()
    coefficients = np.linalg.lstsq(
        calendar_features(train.index)[observed], train.Temperature.to_numpy()[observed], rcond=None
    )[0]
    for column, horizon in enumerate(horizons):
        times = dates + pd.Timedelta(hours=horizon)
        predictions["daily_seasonal"][eligible, column] = frame.Temperature.reindex(
            times - pd.Timedelta(hours=24)
        ).to_numpy()
        predictions["harmonic_climatology"][eligible, column] = calendar_features(times) @ coefficients
    fit = dict(
        fit_start=str(frame.index[boundary - training]),
        fit_end_exclusive=str(quarter),
        last_label_observation=str(frame.index[rows[-1] + max(horizons)]),
        training_rows=len(rows),
        purge_hours=max(horizons),
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **predictions)
    path.with_suffix(".json").write_text(json.dumps(state, indent=2) + "\n")
    path.with_suffix(".fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    temporary.replace(path)
    return predictions, fit
