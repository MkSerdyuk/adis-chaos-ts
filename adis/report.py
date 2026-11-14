import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import run_dlinear
from .evaluation import compare_masks, errors, horizon_metrics
from .experiment import _plot, _published_comparison


def table_audit():
    frame = pd.read_csv("references/published_tables.csv")
    inconsistencies = []
    for row in frame.itertuples():
        p, r, f = row.paper_precision, row.paper_recall, row.paper_f1
        if not np.isfinite([p, r, f]).all() or p + r == 0:
            continue
        low = [max(0, p - 0.005), max(0, r - 0.005)]
        high = [min(1, p + 0.005), min(1, r + 0.005)]
        minimum = 2 * low[0] * low[1] / sum(low) if sum(low) else 0
        maximum = 2 * high[0] * high[1] / sum(high)
        if not minimum - 0.005 <= f <= maximum + 0.005:
            inconsistencies.append(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "horizon": row.horizon,
                    "published_f1": f,
                    "implied_f1": 2 * p * r / (p + r),
                }
            )
    oracle_contradictions = []
    for row in frame[frame.method == "cl_avg_ap"].itertuples():
        if row.rmse > 0.05:
            oracle_contradictions.append(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "horizon": row.horizon,
                    "published_rmse": row.rmse,
                    "oracle_error_bound_assumed": 0.05,
                }
            )
    numeric = frame[["np_percent", "mape", "rmse", "paper_precision", "paper_recall", "paper_f1"]]
    return {
        "rows": len(frame),
        "numeric_cells": int(numeric.notna().sum().sum()),
        "total_slots": numeric.size,
        "f1_inconsistencies_beyond_rounding": inconsistencies,
        "oracle_rmse_above_threshold": oracle_contradictions,
        "oracle_note": "conditional on epsilon=0.05; original oracle tolerance is not uniquely specified",
        "precision_recall_note": "swapped predictable-set convention is inferred for cl rows; trajectory masks remain unresolved",
    }


def rebuild(directory, add_dlinear=False):
    metadata = json.loads((directory / "metadata.json").read_text())
    split = np.load(directory / "split.npz")
    truth = split["truth"]
    scale, minimum = metadata["normalization"]["range"], metadata["normalization"]["minimum"]
    predictions = {p.stem: np.load(p) for p in sorted(directory.glob("*.npy"))}
    if add_dlinear:
        prediction, trials = run_dlinear(
            split["series"],
            metadata["train"][1],
            split["validation_origins"],
            split["test_origins"],
            metadata["horizon"],
        )
        predictions["dlinear_ridge"] = prediction
        np.save(directory / "dlinear_ridge.npy", prediction)
        pd.DataFrame(trials).to_csv(directory / "dlinear_validation.csv", index=False)
    for name, prediction in predictions.items():
        if prediction.shape != truth.shape or np.isinf(prediction).any():
            raise ValueError(f"Invalid prediction artifact: {name}")
    metrics = pd.concat(
        [
            horizon_metrics(
                truth, prediction, name, predictions["trj_avg_ap" if name.startswith("trj_") else "cl_avg_ap"]
            )
            for name, prediction in predictions.items()
        ],
        ignore_index=True,
    )
    metrics["rmse_original_units"] = metrics.rmse * scale
    metrics["mae_original_units"] = metrics.mae * scale
    metrics.to_csv(directory / "metrics.csv", index=False)
    rows = []
    for name, prediction in predictions.items():
        row = {"method": name} | errors(truth, prediction)
        original = errors(truth * scale + minimum, prediction * scale + minimum)
        row.update(
            {
                "rmse_original_units": original["rmse"],
                "mae_original_units": original["mae"],
                "mape_original_percent": original["mape_percent"],
            }
        )
        rows.append(row)
    summaries = pd.DataFrame(rows)
    summaries.to_csv(directory / "summary.csv", index=False)
    selective = {
        name: prediction
        for name, prediction in predictions.items()
        if name.startswith(("cl_", "trj_")) and not name.endswith("_ap")
    }
    masks = compare_masks(truth, predictions, selective)
    if not (masks["count"] == masks.selected_count).all():
        raise AssertionError("A matched-mask comparison changed the accepted denominator")
    masks["rmse_original_units"] = masks.rmse * scale
    masks.to_csv(directory / "matched_masks.csv", index=False)
    comparison = _published_comparison(directory, metadata["dataset"], metrics)
    metadata.pop("table_cells_checked", None)
    metadata["table_slots"] = len(comparison)
    metadata["published_numeric_cells"] = int(comparison.published.notna().sum())
    metadata["comparable_cells"] = int((comparison.status == "comparable_reconstruction").sum())
    metadata["within_published_rounding"] = int(comparison.within_rounding.sum())
    metadata["methods"] = sorted(predictions)
    metadata["forecast_pairs"] = int(np.isfinite(truth).sum())
    metadata["report_source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("adis").glob("*.py"))
    }
    metadata.setdefault("execution_source_sha256", None)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    expected = {"lorenz": (0.68, 0.45, 0.92), "germany": (0.499, 0.372, 0.125)}[metadata["dataset"]]
    measured = [*diagnostics["entropy_complexity"], max(diagnostics["lyapunov_spectrum"])]
    diagnostic_rows = [
        {
            "quantity": name,
            "published": p,
            "measured": m,
            "absolute_difference": abs(p - m),
            "status": "estimator_underdetermined",
        }
        for name, p, m in zip(["entropy", "complexity", "highest_lyapunov"], expected, measured, strict=True)
    ]
    if metadata["dataset"] == "lorenz":
        for name, published in [("time", 7.5), ("steps", 75)]:
            measured_value = diagnostics["paper_horizon_arithmetic"][name]
            diagnostic_rows.append(
                {
                    "quantity": f"horizon_{name}",
                    "published": published,
                    "measured": measured_value,
                    "absolute_difference": abs(published - measured_value),
                    "status": "heuristic_arithmetic_within_published_rounding",
                }
            )
    pd.DataFrame(diagnostic_rows).to_csv(directory / "diagnostic_comparison.csv", index=False)
    _plot(directory, metrics)
    example_names = [
        name
        for name in ["cl_avg_cm", "cl_avg_lr", "trj_trjp_rgdbscan", "dlinear_ridge", "svr_recursive"]
        if name in predictions
    ]
    figure, axes = plt.subplots(
        len(example_names), 1, figsize=(10, 2.5 * len(example_names)), sharex=True, squeeze=False
    )
    physical = metadata["dataset"] == "germany"
    multiplier, offset = (scale, minimum) if physical else (1, 0)
    lead = np.arange(1, truth.shape[1] + 1)
    example_row = np.flatnonzero(np.isfinite(truth).all(axis=1))[0]
    for axis, name in zip(axes[:, 0], example_names, strict=True):
        axis.plot(lead, truth[example_row] * multiplier + offset, color="black", label="Actual")
        axis.plot(lead, predictions[name][example_row] * multiplier + offset, ".--", markersize=3, label=name)
        axis.set_ylabel("Load (MW)" if physical else "Normalized x")
        axis.set_title(
            f"Accepted: {np.isfinite(predictions[name][example_row]).sum()}/{truth.shape[1]}", fontsize=9
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Lead time (hours)" if physical else "Lead time (steps)")
    figure.tight_layout()
    figure.savefig(directory / "example.png", dpi=160)
    figure.savefig(directory / "example.pdf")
    plt.close(figure)
    return metadata, summaries, metrics


def _table(rows, scale=1):
    lines = ["| Метод | Покрытие | RMSE |", "|---|---:|---:|"]
    for row in rows.itertuples():
        lines.append(f"| `{row.method}` | {row.coverage:.1%} | {row.rmse * scale:.3f} |")
    return "\n".join(lines)


def write_report(root, all_results):
    report = [
        "# Проверка статьи и сравнение прогнозов",
        "",
        "Численные таблицы статьи **не воспроизведены с точностью округления**. "
        "Выполнена вычислимая реконструкция метода с явно заданными недостающими настройками. "
        "Результаты ниже получены запуском Python, все прогнозы сохранены.",
        "",
    ]
    for metadata, summaries, metrics in all_results:
        dataset = metadata["dataset"]
        germany = dataset == "germany"
        report += [
            f"## {'Немецкая нагрузка' if germany else 'Лоренц'}",
            "",
            f"Train: {metadata['train'][1]} наблюдений; validation: 1000; test: 1000. "
            f"Общие {metadata['test_origins']} origins, горизонт {metadata['horizon']}. "
            "Одна и та же пара (origin, horizon) используется каждым методом. "
            f"Оценивается {metadata['forecast_pairs']} пар; соседние окна перекрываются "
            "и не являются независимыми наблюдениями.",
            "",
        ]
        if metadata.get("evaluation") == "paper":
            counts = metadata["targets_per_horizon"]
            sample_size = str(min(counts)) if min(counts) == max(counts) else f"от {min(counts)} до {max(counts)}"
            report += [
                f"По формуле (7), на каждом горизонте оценивается {sample_size} "
                "точек test (при test_stride=1 — весь test). "
                "Origin равен target − horizon; значения вне test исключены из метрик. "
                "Для AP доступна отдельная полная матрица истинных значений, включая "
                "промежуточные точки перед началом test.",
                f"Последние {metadata['validation_embargo']} значений validation исключены из "
                "обучения классификаторов и настройки всех моделей: калибровка завершается "
                "не позже самого раннего тестового origin.",
                "",
            ]
        if germany:
            matched = pd.read_csv(root / dataset / "matched_masks.csv")
            matched = matched[matched.selection_method == "trj_trjp_rgdbscan"].set_index("method")
            baseline, partial = matched.loc["svr_recursive"], matched.loc["trj_trjp_rgdbscan"]
            report += [
                f"На одной и той же маске из {int(partial.selected_count)} принятых пар "
                f"SVR имеет RMSE {baseline.rmse_original_units:.1f} МВт, "
                f"а `trj_trjp_rgdbscan` — {partial.rmse_original_units:.1f} МВт. "
                "Это результат нашей реконструкции, а не точное воспроизведение авторского эксперимента. "
                "SVR использует историю 168 часов и календарь; метод статьи — историю 30 часов.",
                "",
            ]
        if germany:
            report += [
                "OPSD 2020-10-06, `DE_load_actual_entsoe_transparency`, 2015-01-01 — 2016-02-20, UTC. "
                "Начальная запись 2014-12-31T23:00Z пустая; интервал явно начинается со следующего часа. "
                "Пропуски не заполнялись. Источник исходного немецкого ряда авторы не указали.",
                "",
            ]
        full = summaries[summaries.coverage == 1].sort_values("rmse")
        selected_names = ["cl_avg_cm", "cl_avg_lr", "cl_avg_rgdbscan", "trj_trjp_rgdbscan", "cl_avg_ap"]
        selected = summaries[summaries.method.isin(selected_names)]
        report += [
            f"RMSE в {'МВт' if germany else 'единицах нормированного ряда'}. "
            "Ошибка усреднена по всем горизонтам 1–100 и всем принятым прогнозам. "
            "`ap` использует будущее истинное значение и служит только оракулом.",
            "",
            _table(
                pd.concat([full, selected]).drop_duplicates("method"),
                metadata["normalization"]["range"] if germany else 1,
            ),
            "",
        ]
        horizon_rows = metrics[
            metrics.method.isin(["cl_avg_cm", "trj_trjp_rgdbscan", "seasonal_168", "ridge", "dlinear_ridge"])
            & metrics.horizon.isin([1, 10, 24, 50, 100])
        ]
        horizon_rows.to_csv(root / dataset / "selected_horizons.csv", index=False)
        report += [
            f"Из {metadata['published_numeric_cells']} числовых ячеек статьи "
            f"сравнимы {metadata['comparable_cells']}; "
            f"в пределах округления оказались {metadata['within_published_rounding']}. "
            "Совпадение отдельных ячеек не подтверждает репликацию: входы и настройки авторов не восстановлены.",
            "",
            f"![Ошибка и покрытие]({dataset}/horizons.png)",
            "",
            f"Первое тестовое окно, выбранное без отбора по качеству: ![Пример прогноза]({dataset}/example.png)",
            "",
        ]
    report += [
        "## Как читать сравнение",
        "",
        "`metrics.csv` содержит кривые каждого метода по горизонтам, `summary.csv` — сводку. "
        "В `matched_masks.csv` каждый обычный прогнозировщик оценивается на точной маске "
        "соответствующего частичного метода; `selected_count` и `count` совпадают. "
        "Сравнения разных частичных методов на пересечениях масок исключены.",
        "",
        "Мотивы, нормализация и регрессоры обучаются только на train. "
        "Validation используется для SMOTE/классификаторов предсказуемости, порога энтропии "
        "и выбора параметров регрессоров. Test не используется для настройки. "
        "Внутри окна используются рекурсивные или прямые прогнозы; будущие наблюдения доступны только AP. "
        "При переходе к следующему origin ранее случившиеся наблюдения доступны всем методам.",
        "",
        "Метод статьи получает 30 прошлых значений, `ridge_30` — те же 30. "
        "Сезонные модели и остальные регрессоры используют историю до 168 часов; "
        "для Германии регрессоры также получают известный календарь. "
        "Они решают тот же тест на тех же датах с более широким контекстом; "
        "это отличие учитывается при интерпретации.",
        "",
        "## Ограничения точной репликации",
        "",
        "Не указаны начальная точка и координата Лоренца, конкретная нормализация, "
        "размер и положение validation, происхождение и нарезка немецкого ряда, "
        "параметры DBSCAN, число и шум траекторий, часть настроек классификаторов. "
        "Германия в статье содержит около 10 тысяч часов всего, поэтому нарезка "
        "10 000 train + 1000 test для неё невозможна. Наши допущения перечислены в README и metadata.json.",
        "",
        "В таблице 4 для `trj/avg/cm`, h=50, P=1 и R=0.39 дают F1≈0.561; "
        "опубликовано 0.65. Это расхождение превосходит округление. "
        "Для строк `cl` значения P/R согласуются с переставленными precision/recall "
        "предсказуемых точек. Для ряда `trj` даже эта интерпретация не согласуется "
        "с покрытием семейного оракула; использованная авторами маска неизвестна. "
        "Выгружены стандартные метрики для обеих категорий и отдельно предполагаемая конвенция статьи.",
        "",
        "AP отбрасывает точки с |ошибка|≥ε, поэтому его RMSE обязана быть <ε. "
        "Часть опубликованных AP-RMSE превышает 0.05 и 0.01; "
        "воспроизвести их одновременно с описанным правилом при этих порогах нельзя. "
        "Статья не даёт однозначного отдельного порога AP. Полный аудит — `table_audit.json`.",
        "",
        "Диагностические entropy/complexity и спектр Э́кмана сохранены отдельно. "
        "Параметры оценивания в статье отсутствуют; совпадение её значений не заявляется. "
        "75.084 шага — проверенная арифметика авторской эвристики, а не доказанная граница прогнозируемости.",
        "",
        "Оценка Э́кмана для нашего ряда Лоренца независимо совпала с upstream "
        "[`nolds.lyap_e`](https://raw.githubusercontent.com/CSchoel/nolds/master/nolds/measures.py) "
        "с максимальной разницей спектров 5.42×10⁻¹⁴. Это подтверждает соответствие реализации "
        "при выбранных настройках, а не точность восстановления истинного показателя системы. "
        "Подробности независимых сверок — `references/verification.json`.",
        "",
        "## Источники дополнительных моделей",
        "",
        "[Wang et al., 2025](https://www.mdpi.com/1999-4893/18/11/695) "
        "используют тот же выпуск OPSD и семейства kNN, деревьев и нейросетей. "
        "[Aksoy, 2026](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rpg2.70378) "
        "сравнивает сезонные модели и DLinear на европейских сетях, включая немецкую. "
        "Здесь эти семейства переобучены на нашей общей нарезке; чужие опубликованные "
        "оценки не выдаются за результаты сопоставимого теста. `dlinear_ridge` сохраняет "
        "две линейные головы разложения DLinear, но обучает их совместно регуляризованным "
        "методом наименьших квадратов вместо Adam; это обозначенная модификация.",
        "",
        "## Повторная проверка открытого кода",
        "",
        "12 ноября 2026 года повторены веб-поиск, поиск GitHub API и проверка исходников. "
        "Публичные сторонние реализации есть: например, "
        "[MaruwanDono/PredictingChaoticTimeSeries](https://github.com/MaruwanDono/PredictingChaoticTimeSeries) "
        "и [Shuflin80/Chaotic-timeseries-prediction](https://github.com/Shuflin80/Chaotic-timeseries-prediction). "
        "Дополнительно проверены связанные ВКР Antonova и Tebiev. Подтверждённого авторского кода "
        "с исходными данными и настройками таблиц 2023 года не найдено. Перечень, версии и "
        "ограничения сохранены в `references/source_audit.json`; ссылки и краткий разбор — в README. "
        "Отсутствие других материалов этим поиском не доказано.",
        "",
        "[Исходная статья Gromov & Baranov, 2023](https://onlinelibrary.wiley.com/doi/10.1155/2023/6689371). "
        "[Доступная копия PDF в ВШЭ](https://publications.hse.ru/pubs/share/direct/861165160.pdf).",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/fidelity")
    parser.add_argument("--add-dlinear", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    results = [rebuild(root / dataset, args.add_dlinear) for dataset in ["lorenz", "germany"]]
    (root / "table_audit.json").write_text(json.dumps(table_audit(), indent=2) + "\n")
    write_report(root, results)
    for directory in [root / "lorenz", root / "germany"]:
        artifacts = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())
            if p.suffix in {".npy", ".npz", ".csv", ".json", ".png", ".pdf"} and p.name != "checksums.json"
        }
        (directory / "checksums.json").write_text(json.dumps(artifacts, indent=2) + "\n")


if __name__ == "__main__":
    main()
