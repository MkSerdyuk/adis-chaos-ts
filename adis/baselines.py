import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .evaluation import errors

LAGS = np.r_[np.arange(30), 47, 71, 167]


def _calendar(origins, timestamps):
    dates = timestamps[origins].tz_convert("Europe/Berlin")
    hour = dates.hour.to_numpy()
    week = dates.day_of_week.to_numpy() * 24 + hour
    return np.c_[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * week / 168),
        np.cos(2 * np.pi * week / 168),
    ]


def _features(series, origins, lags=LAGS, timestamps=None):
    features = series[origins[:, None] - lags]
    if timestamps is not None:
        features = np.c_[features, _calendar(origins, timestamps)]
    return features


def seasonal_naive(series, origins, horizon, period):
    offsets = (np.arange(horizon) % period) - period + 1
    return series[origins[:, None] + offsets]


def _recursive(model, series, origins, horizon, timestamps):
    buffer = series[origins[:, None] - np.arange(167, -1, -1)].copy()
    result = np.empty((len(origins), horizon))
    for h in range(horizon):
        features = buffer[:, -1 - LAGS]
        if timestamps is not None:
            calendar = _calendar(origins + h, timestamps)
            features = np.c_[features, calendar]
        result[:, h] = model.predict(features)
        buffer = np.c_[buffer[:, 1:], result[:, h]]
    return result


def run_baselines(series, train_end, validation_origins, test_origins, horizon, timestamps=None, seed=42):
    target_offsets = np.arange(1, horizon + 1)
    val_truth = series[validation_origins[:, None] + target_offsets]
    predictions = {
        "persistence": seasonal_naive(series, test_origins, horizon, 1),
        "seasonal_24": seasonal_naive(series, test_origins, horizon, 24),
        "seasonal_168": seasonal_naive(series, test_origins, horizon, 168),
    }
    trials = []
    specifications = {
        "ridge_30": (False, np.arange(30), [Ridge(alpha=a) for a in (0.1, 10, 100)]),
        "ridge": (False, LAGS, [Ridge(alpha=a) for a in (0.1, 10, 100)]),
        "knn": (False, LAGS, [KNeighborsRegressor(n_neighbors=k, weights="distance") for k in (3, 10)]),
        "decision_tree": (
            False,
            LAGS,
            [DecisionTreeRegressor(max_depth=d, min_samples_leaf=5, random_state=seed) for d in (8, 14)],
        ),
        "random_forest": (
            False,
            LAGS,
            [
                RandomForestRegressor(
                    n_estimators=80, max_depth=16, min_samples_leaf=leaf, n_jobs=4, random_state=seed
                )
                for leaf in (3, 10)
            ],
        ),
        "mlp": (
            False,
            LAGS,
            [MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, random_state=seed)],
        ),
        "svr_recursive": (True, LAGS, [SVR(C=c, epsilon=0.005) for c in (1, 10)]),
        "hist_gradient_boosting_recursive": (
            True,
            LAGS,
            [
                HistGradientBoostingRegressor(
                    max_iter=150,
                    max_leaf_nodes=leaves,
                    l2_regularization=1,
                    random_state=seed,
                    early_stopping=False,
                )
                for leaves in (15, 31)
            ],
        ),
    }
    for name, (recursive, lags, candidates) in specifications.items():
        start = time.monotonic()
        train_origins = np.arange(167, train_end - (1 if recursive else horizon))
        x = _features(series, train_origins, lags, timestamps)
        y = series[train_origins + 1] if recursive else series[train_origins[:, None] + target_offsets]
        best, best_error = None, np.inf
        for candidate in candidates:
            model = make_pipeline(StandardScaler(), candidate).fit(x, y)
            if recursive:
                val = _recursive(model, series, validation_origins, horizon, timestamps)
            else:
                val = model.predict(_features(series, validation_origins, lags, timestamps))
            score = errors(val_truth, val)["rmse"]
            trials.append({"method": name, "parameters": str(candidate), "validation_rmse": score})
            if score < best_error:
                best, best_error = model, score
        if recursive:
            predictions[name] = _recursive(best, series, test_origins, horizon, timestamps)
        else:
            predictions[name] = best.predict(_features(series, test_origins, lags, timestamps))
        print(
            f"  baseline {name}: validation RMSE {best_error:.6f}, {time.monotonic() - start:.1f}s",
            flush=True,
        )
    dlinear, dlinear_trials = run_dlinear(series, train_end, validation_origins, test_origins, horizon)
    predictions["dlinear_ridge"] = dlinear
    trials.extend(dlinear_trials)
    return predictions, trials


def run_dlinear(series, train_end, validation_origins, test_origins, horizon):
    def features(origins):
        history = series[origins[:, None] - np.arange(167, -1, -1)]
        trend = uniform_filter1d(history, size=25, axis=1, mode="nearest")
        return np.c_[history - trend, trend]

    origins = np.arange(167, train_end - horizon)
    offsets = np.arange(1, horizon + 1)
    x, y = features(origins), series[origins[:, None] + offsets]
    validation = features(validation_origins)
    truth = series[validation_origins[:, None] + offsets]
    best, best_error = None, np.inf
    trials = []
    for alpha in (0.1, 10, 100):
        model = Ridge(alpha=alpha).fit(x, y)
        score = errors(truth, model.predict(validation))["rmse"]
        trials.append(
            {"method": "dlinear_ridge", "parameters": f"window=25, alpha={alpha}", "validation_rmse": score}
        )
        if score < best_error:
            best, best_error = model, score
    print(f"  baseline dlinear_ridge: validation RMSE {best_error:.6f}", flush=True)
    return best.predict(features(test_origins)), trials
