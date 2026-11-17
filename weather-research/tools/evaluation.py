import numpy as np
import pandas as pd

from .baselines import NAMES


def scores(truth, prediction):
    errors = np.asarray(prediction) - np.asarray(truth)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return dict(count=0, mae=np.nan, rmse=np.nan, bias=np.nan, tail95=np.nan)
    absolute = np.abs(errors)
    count = max(1, int(np.ceil(0.05 * len(errors))))
    return dict(
        count=len(errors),
        mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(errors**2))),
        bias=float(errors.mean()),
        tail95=float(np.sort(absolute)[-count:].mean()),
    )


def metrics(predictions, periods):
    records = []
    for period, (start, end) in periods.items():
        subset = predictions[
            (predictions.origin >= pd.Timestamp(start, tz="UTC"))
            & (predictions.origin < pd.Timestamp(end, tz="UTC"))
            & (predictions.target_time < pd.Timestamp(end, tz="UTC"))
        ]
        for (station, horizon), group in subset.groupby(["station", "horizon"], sort=False):
            available = group[group.eligible & group.truth.notna()]
            denominator = available.origin.nunique()
            origins = group.drop_duplicates("origin")
            availability = dict(
                scheduled_targets=len(origins),
                missing_truth_targets=int(origins.truth.isna().sum()),
                missing_input_targets=int((~origins.eligible & origins.truth.notna()).sum()),
            )
            assert sum(
                availability[key] for key in ["missing_truth_targets", "missing_input_targets"]
            ) + denominator == len(origins)
            for name, rows in available.groupby("method", sort=False):
                values = scores(rows.truth, rows.prediction)
                records.append(
                    dict(
                        period=period,
                        station=station,
                        horizon=horizon,
                        mask="available",
                        method=name,
                        eligible_targets=denominator,
                        coverage=values["count"] / denominator if denominator else np.nan,
                        **availability,
                        **values,
                    )
                )
            for reference in ["forecaster", "paper_geometry"]:
                accepted = available[(available.method == reference) & available.prediction.notna()].origin
                paired = available[
                    available.origin.isin(accepted) & available.method.isin([reference, *NAMES])
                ]
                for name in [reference, *NAMES]:
                    rows = paired[paired.method == name]
                    values = scores(rows.truth, rows.prediction)
                    assert values["count"] == len(accepted)
                    records.append(
                        dict(
                            period=period,
                            station=station,
                            horizon=horizon,
                            mask=reference,
                            method=name,
                            eligible_targets=denominator,
                            coverage=len(accepted) / denominator if denominator else np.nan,
                            **availability,
                            **values,
                        )
                    )
    return pd.DataFrame(records)


def bootstrap(predictions, stations, periods, repetitions=10000, seed=20261117):
    records = []
    methods = ["forecaster", *NAMES]
    for period, (start, end) in periods.items():
        if period == "validation":
            continue
        days = pd.date_range(start, end, freq="D", inclusive="left", tz="UTC") + pd.Timedelta(hours=23)
        errors = np.full((len(methods), len(stations), len(days)), np.nan)
        subset = predictions[
            (predictions.horizon == 24) & (predictions.target_time < pd.Timestamp(end, tz="UTC"))
        ]
        for j, station in enumerate(stations):
            rows = subset[subset.station == station]
            reference = rows[rows.method == "forecaster"].set_index("origin").reindex(days)
            accepted = reference.prediction.notna() & reference.truth.notna() & reference.eligible.eq(True)
            for i, method in enumerate(methods):
                values = rows[rows.method == method].set_index("origin").reindex(days)
                assert values.prediction.loc[accepted].notna().all()
                errors[i, j, accepted.to_numpy()] = np.abs(
                    values.prediction.loc[accepted] - values.truth.loc[accepted]
                )
        counts = np.isfinite(errors).sum(axis=2)
        means = np.divide(
            np.nansum(errors, axis=2), counts, out=np.full(counts.shape, np.nan), where=counts > 0
        )
        observed = (means[0] - means[1:]).mean(axis=1)
        for block in [7, 14, 28]:
            rng = np.random.default_rng(seed)
            samples = np.empty((repetitions, len(NAMES)))
            for first in range(0, repetitions, 250):
                size = min(250, repetitions - first)
                starts = rng.integers(0, len(days), size=(size, int(np.ceil(len(days) / block))))
                indices = ((starts[:, :, None] + np.arange(block)) % len(days)).reshape(size, -1)[
                    :, : len(days)
                ]
                selected = np.take(errors, indices, axis=2)
                count = np.isfinite(selected).sum(axis=3)
                mean = np.divide(
                    np.nansum(selected, axis=3), count, out=np.full(count.shape, np.nan), where=count > 0
                )
                samples[first : first + size] = (mean[0] - mean[1:]).mean(axis=1).T
            lower, upper = np.quantile(samples, [0.025 / len(NAMES), 1 - 0.025 / len(NAMES)], axis=0)
            for i, method in enumerate(NAMES):
                records.append(
                    dict(
                        period=period,
                        horizon=24,
                        comparator=method,
                        block_days=block,
                        primary=period == "holdout" and block == 14,
                        difference=observed[i],
                        lower=lower[i],
                        upper=upper[i],
                        family_size=len(NAMES),
                        interval_confidence=1 - 0.05 / len(NAMES),
                        replications=repetitions,
                        seed=seed,
                        station_weights="equal",
                        accepted_per_station=counts[0].tolist(),
                        finite_replicates=int(np.isfinite(samples[:, i]).sum()),
                    )
                )
    return pd.DataFrame(records)
