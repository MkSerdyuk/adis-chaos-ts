import numpy as np
import pandas as pd


def errors(truth, prediction, mask=None):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    evaluated = np.isfinite(truth)
    total = int(evaluated.sum())
    if not total:
        raise ValueError("No evaluation targets")
    accepted = evaluated & np.isfinite(prediction)
    if mask is not None:
        accepted &= mask
    count = int(accepted.sum())
    result = {
        "count": count,
        "total": total,
        "coverage": count / total,
        "np_percent": 100 * (1 - count / total),
    }
    if not count:
        return result | {k: np.nan for k in ["mae", "rmse", "mape", "mape_percent"]} | {"i2": 0.0}
    delta = prediction[accepted] - truth[accepted]
    mae = np.abs(delta).mean()
    mape = np.mean(np.abs(delta / truth[accepted])) if np.all(truth[accepted] != 0) else np.nan
    return result | {
        "mae": mae,
        "rmse": np.sqrt(np.mean(delta**2)),
        "mape": mape,
        "mape_percent": 100 * mape,
        "i2": np.abs(delta).sum() / total,
    }


def mask_metrics(prediction, oracle):
    predicted, actual = np.isfinite(prediction), np.isfinite(oracle)
    result = {}
    for label, a, b in [("predictable", predicted, actual), ("np", ~predicted, ~actual)]:
        tp, fp, fn = np.sum(a & b), np.sum(a & ~b), np.sum(~a & b)
        result[f"{label}_precision"] = tp / (tp + fp) if tp + fp else np.nan
        result[f"{label}_recall"] = tp / (tp + fn) if tp + fn else np.nan
        result[f"{label}_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    result["i3"] = int(np.sum(predicted != actual))
    result["paper_precision"] = result["predictable_recall"]
    result["paper_recall"] = result["predictable_precision"]
    result["paper_f1"] = result["predictable_f1"]
    return result


def horizon_metrics(truth, prediction, name, oracle=None):
    rows = []
    for h in range(truth.shape[1]):
        row = {"method": name, "horizon": h + 1} | errors(truth[:, h], prediction[:, h])
        if oracle is not None:
            evaluated = np.isfinite(truth[:, h])
            row |= mask_metrics(prediction[evaluated, h], oracle[evaluated, h])
        rows.append(row)
    return pd.DataFrame(rows)


def compare_masks(truth, predictions, selective):
    rows = []
    for selective_name, selected in selective.items():
        for name, prediction in predictions.items():
            if not np.isfinite(prediction[np.isfinite(truth)]).all() and name != selective_name:
                continue
            rows.append(
                {
                    "selection_method": selective_name,
                    "method": name,
                    "selected_count": int((np.isfinite(selected) & np.isfinite(truth)).sum()),
                }
                | errors(truth, prediction, np.isfinite(selected))
            )
    return pd.DataFrame(rows)
