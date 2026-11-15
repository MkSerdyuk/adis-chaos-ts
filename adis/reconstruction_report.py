import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import errors, horizon_metrics
from .reconstruction import PROFILES, check_frozen


def table(frame):
    def cell(value):
        if isinstance(value, float):
            return f"{value:.5f}" if np.isfinite(value) else "—"
        return str(value)

    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.values.tolist()]
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows)


def build(root):
    paper = pd.read_csv("references/published_tables.csv")
    paper = paper[paper.method == "trj_trjp_rgdbscan"]
    all_summaries, all_metrics, all_comparisons, common, selected, diagnostics, baseline_comparisons = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for profile in PROFILES:
        directory = root / profile
        check_frozen(directory)
        metadata = json.loads((directory / "data.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        diag = json.loads((directory / "diagnostics.json").read_text())
        selected.append(
            {
                "profile": profile,
                "trial": selection["trial"],
                **{
                    k: selection["config"][k]
                    for k in ["epsilon", "cluster_eps", "min_samples", "perturbation_variance"]
                },
            }
        )
        diagnostics.append(
            {
                "profile": profile,
                "H": diag["entropy_complexity"][0],
                "C": diag["entropy_complexity"][1],
                "lambda_max": diag["lyapunov_spectrum"][0],
            }
        )
        summaries, metrics = [], []
        for partition in ["test", "fresh"]:
            targets = np.load(directory / f"{partition}_targets.npz")
            truth = targets["truth"]
            np.testing.assert_array_equal(np.isfinite(truth).sum(axis=0), np.full(100, 1000))
            for name in ["baseline", "selected"]:
                prediction = np.load(directory / f"{partition}_{name}.npy")
                oracle = np.load(directory / f"{partition}_{name}_ap.npy")
                summaries.append(
                    {"profile": profile, "partition": partition, "method": name} | errors(truth, prediction)
                )
                metrics.append(
                    horizon_metrics(truth, prediction, name, oracle).assign(
                        profile=profile, partition=partition
                    )
                )
        summary, measured = pd.DataFrame(summaries), pd.concat(metrics)
        summary["rmse_original_units"] = summary.rmse * metadata["normalization"]["range"]
        all_summaries.append(summary)
        all_metrics.append(measured)
        common.append(pd.read_csv(directory / "intersection.csv").assign(profile=profile))
        baseline_summary = pd.read_csv(directory / "baseline_summary.csv")
        matched = pd.read_csv(directory / "matched_masks.csv")
        for partition in ["test", "fresh"]:
            assert baseline_summary[baseline_summary.partition == partition].method.nunique() == 12
            candidates = matched[(matched.partition == partition) & (matched.method != "selected")]
            best = candidates.sort_values("rmse").iloc[0]
            ours = matched[(matched.partition == partition) & (matched.method == "selected")].iloc[0]
            assert int(best["count"]) == int(ours["count"]) == int(ours.selected_count)
            baseline_comparisons.append(
                {
                    "profile": profile,
                    "partition": partition,
                    "baseline": best.method,
                    "baseline_rmse": best.rmse,
                    "selected_rmse": ours.rmse,
                    "coverage": ours.coverage,
                }
            )
        dataset = "germany" if profile.startswith("germany") else "lorenz"
        joined = measured.merge(paper[paper.dataset == dataset], on="horizon", suffixes=("", "_paper"))
        for _, row in joined.iterrows():
            for metric in ["np_percent", "rmse", "mape", "paper_precision", "paper_recall", "paper_f1"]:
                expected, observed = row[f"{metric}_paper"], row[metric]
                tolerance = 0.5 if metric == "np_percent" else 0.005
                all_comparisons.append(
                    {
                        "profile": profile,
                        "partition": row.partition,
                        "method": row.method,
                        "horizon": row.horizon,
                        "metric": metric,
                        "paper": expected,
                        "measured": observed,
                        "absolute_difference": abs(expected - observed),
                        "rounding_tolerance": tolerance,
                        "status": "comparable"
                        if np.isfinite(expected) and np.isfinite(observed)
                        else "undefined",
                        "within_rounding": bool(abs(expected - observed) <= tolerance),
                    }
                )
    summary, metrics = pd.concat(all_summaries), pd.concat(all_metrics)
    h100 = metrics[
        (metrics.profile == "lorenz_rk4") & (metrics.partition == "test") & (metrics.horizon == 100)
    ].set_index("method")
    comparisons, common = pd.DataFrame(all_comparisons), pd.concat(common)
    matches = (
        comparisons[(comparisons.partition == "test") & (comparisons.method == "selected")]
        .groupby("profile", sort=False)
        .agg(
            published_cells=("metric", "size"),
            finite_comparisons=("status", lambda values: (values == "comparable").sum()),
            within_rounding=("within_rounding", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(root / "summary.csv", index=False)
    metrics.to_csv(root / "metrics.csv", index=False)
    comparisons.to_csv(root / "paper_comparison.csv", index=False)
    matches.to_csv(root / "paper_match_counts.csv", index=False)
    common.to_csv(root / "intersection.csv", index=False)
    pd.DataFrame(selected).to_csv(root / "selected_parameters.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(root / "diagnostics.csv", index=False)
    pd.DataFrame(baseline_comparisons).to_csv(root / "best_baseline_on_selected_mask.csv", index=False)
    for partition in ["test", "fresh"]:
        fig, axes = plt.subplots(2, len(PROFILES), figsize=(16, 6), sharex=True)
        for column, profile in enumerate(PROFILES):
            subset = metrics[(metrics.profile == profile) & (metrics.partition == partition)]
            for name in ["baseline", "selected"]:
                values = subset[subset.method == name]
                axes[0, column].plot(values.horizon, values.rmse, label=name)
                axes[1, column].plot(values.horizon, values.coverage, label=name)
            published = paper[paper.dataset == ("germany" if profile.startswith("germany") else "lorenz")]
            if partition == "test":
                axes[0, column].scatter(
                    published.horizon, published.rmse, color="black", marker="x", label="paper"
                )
                axes[1, column].scatter(
                    published.horizon, 1 - published.np_percent / 100, color="black", marker="x"
                )
            axes[0, column].set_title(profile)
            axes[1, column].set_xlabel("Horizon")
            axes[1, column].set_ylim(0, 1.02)
            for ax in axes[:, column]:
                ax.grid(alpha=0.2)
        axes[0, 0].set_ylabel("Normalized RMSE")
        axes[1, 0].set_ylabel("Coverage")
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(root / f"{partition}.png", dpi=160)
        fig.savefig(root / f"{partition}.pdf")
        plt.close(fig)
    report = [
        "# Повторная реконструкция окружения и подбор параметров",
        "Точное воспроизведение статьи не подтверждено. Все варианты данных опубликованы; лучший источник по test не выбирался.",
        "Этот подбор относится к строке trj/trjp/rgdbscan и сравнению с 12 дополнительными моделями. [Предыдущая сверка формул](../fidelity/AUDIT.md) и [полный эксперимент с остальными сочетаниями](../fidelity/REPORT.md) сохранены отдельно; они не выдаются за повторный запуск всех вариантов после последнего исправления Wishart.",
        "## Что исправлено",
        "В Algorithm 1 слияние сохраняет кластер c1. Удалён дополнительный перенос флага completed из поглощённых кластеров через OR. Это буквальное чтение псевдокода, совпадающее с открытой реализацией Антоновой; исходный код авторов статьи не найден. Все тренировочные мотивы пересчитаны.",
        "Для отдельного оператора spread добавлена калибровка среднего размаха конечных точек на validation по каждому горизонту, умноженного на заданный factor. Возвращается число доступных наблюдений; пустой горизонт вызывает ошибку. Интерпретация spread=ptp и исключение пустых наборов явные. Этот оператор не подменяет rgdbscan и не используется для результатов ниже.",
        "## Данные и подбор",
        "lorenz_rk4 — прежний генератор; lorenz_archive — закреплённый файл из связанной с авторами работы Антоновой, с удалением первых 3000 отсчётов. Нормировка архивного файла заново вычислена только по train. Тождество этого файла входу статьи не установлено.",
        "germany — ENTSO-E Transparency из OPSD2020; germany_power_statistics — другой ряд ENTSO-E Data Portal/Power Statistics из OPSD2018. У Transparency 7984 train, у Power Statistics 8008: последний включает 24 часа 2014-12-31, соответствуя начальной дате статьи (UTC — наше допущение). Даты validation/test/fresh одинаковы: по 1000 наблюдений. Поэтому это не чистая оценка эффекта смены источника. Пропусков нет, даты не подбирались. Выпуски 2018-03-13 и 2018-06-30 Power Statistics идентичны на выбранном интервале; старые выпуски 2016/2017 не покрывают его конец.",
        "Мотивы и нормировка фиксируются по train; наблюдаемая история обновляется до текущего origin. Частота переобучения при перемещении origin в статье не задана. Все этапы и ограничения зафиксированы в protocol.json. 32 пространственные конфигурации на профиль проверены по первому шагу, затем сохранённые кандидаты — по 100 шагам; финалисты проверены на 101 origins и двух seed. Выбор минимизирует средний RMSE по горизонтам при покрытии ≥90% на h1–10, ≥50% в целом и на каждом h50–100 ≥16% для Lorenz / ≥37% для Germany. Первые два порога — наши условия эксперимента, не параметры авторов. Если кандидатов нет, сохраняется исходная настройка.",
        "Validation используется повторно для настройки и не служит независимым подтверждением. Старый test ранее рассматривался. Fresh не участвовал в подборе. На каждом горизонте оцениваются все 1000 целевых точек. Последние 99 наблюдений validation исключены из калибровки. Seed, масштаб и даты не подбирались по test.",
        "## Выбранные параметры",
        table(pd.DataFrame(selected)),
        "Постоянны L=4, Kmax=10, Wishart r=11/μ=.2, λ=.99, 16 траекторий; S не оптимизировалось. Финальный test/fresh оценён с seed 42 и фиксированным правилом seed для батчей. Устойчивость к числу траекторий и множество независимых запусков test не проверялись. Δ=0 допустимо в формуле статьи, но означает одинаковые детерминированные пути и нулевой ансамблевый размах; такой вариант нельзя выдавать за свидетельство эффективности стохастического ансамбля.",
        "В rgdbscan пустой набор кластерных центров даёт неопределённый размах: сравнение не устанавливает его строгий рост и не активирует отказ. При отсутствии конечного UPV точка всё равно отвергается; один центр имеет нулевой размах. Это явное соглашение: статья не задаёт отдельного правила для пустого результата вспомогательной кластеризации. Большой min_samples относительно S может ослаблять этот критерий; причина полного покрытия архивного профиля отдельно по числу кластеров не измерялась.",
        "## Результаты",
        "baseline — прежние гиперпараметры после исправления Wishart; selected — настройки, выбранные только на validation. В сводке RMSE объединяет все принятые пары точка/горизонт; функция выбора усредняет отдельные RMSE по 100 горизонтам. Ошибки условны на принятых точках, поэтому покрытие приводится рядом.",
        "Подбор улучшил средний RMSE обоих рядов Лоренца на test и fresh. Для обоих немецких источников результат отрицательный: покрытие выросло, но RMSE ухудшился на обоих периодах, в том числе на общей маске. Поэтому увеличение числа прогнозов нельзя считать улучшением точности. Эти результаты сохранены без повторной настройки по test.",
        table(summary[["profile", "partition", "method", "rmse", "coverage"]]),
        "![Test](test.png)",
        "![Fresh](fresh.png)",
        "## На общей маске baseline и selected",
        table(common[["profile", "partition", "method", "rmse", "coverage"]]),
        "Снижение среднего RMSE не означает улучшения каждого показателя. Например, у lorenz_rk4 на прежнем test "
        f"при h100 MAPE: {h100.loc['baseline', 'mape']:.5f} → {h100.loc['selected', 'mape']:.5f}. "
        "Все показатели по горизонтам доступны в [metrics.csv](metrics.csv) и [paper_comparison.csv](paper_comparison.csv).",
        "## Сравнение с таблицами",
        table(
            metrics[(metrics.partition == "test") & metrics.horizon.isin([1, 10, 50, 100])][
                ["profile", "method", "horizon", "rmse", "mape", "coverage"]
            ]
        ),
        "В paper_comparison.csv сверены все шесть показателей строки trj/trjp/rgdbscan на h=1,10,50,100. Допуски округления: .005 для ошибок и P/R/F1, .5 процентного пункта для NP. MAPE сопоставляется как доля; конвенция единиц авторов не подтверждена. Fresh — отдельный период, его совпадение не означает воспроизведения статьи. AP — исключительно диагностический оракул с доступом к истине; в подборе и рабочих прогнозах он не участвует. Paper precision/recall используют прежнее явно отмеченное восстановление переставленных обозначений статьи; стандартные метрики сохранены рядом.",
        "Число численных совпадений selected на test с учётом округления приведено ниже. Совпадение отдельных ячеек при неизвестных исходных данных и настройках не подтверждает воспроизведение эксперимента.",
        table(matches),
        "Статья сообщает для Germany RMSE .01/.01/.05/.03 при h=1/10/50/100 и покрытии 100%/100%/37%/49%; для Lorenz .01/.03/.04/.04 и 100%/100%/16%/17%. Это нормированные ошибки, не МВт.",
        "## Диагностика рядов на train",
        table(pd.DataFrame(diagnostics)),
        "Статья: H/C/λ = .68/.45/.92 для Lorenz и .499/.372/.125 для Germany. Наши параметры оценивания указаны в diagnostics.json. Для H/C статья не задаёт даже определение меры. Здесь показана дополнительная ordinal Shannon/MPR-диагностика с D=5: её верхняя граница C около .425, поэтому опубликованное C=.45 она получить не может ни на каком ряде. Наша настройка несовместима с опубликованным числом; причина не установлена: возможны другая мера/настройки либо ошибка публикации. Это не доказывает неверность данных. [Стандартная граница MPR](https://arxiv.org/abs/2102.06786). Оцениватели не подгонялись под опубликованные числа.",
        "## Дополнительные алгоритмы",
        "Для каждого профиля повторены те же 12 baselines на одинаковых границах; baseline_summary.csv хранит полные ошибки, matched_masks.csv — сравнение на маске selected. Регрессоры используют до 168 прошлых значений, некоторые — календарь; метод статьи использует 30, это различие сохранено явно. Все настройки дополнительных моделей выбираются на той же validation, train не расширяется.",
        "Ниже — описательное сравнение с лучшим результатом среди 12 baselines на данной маске. Это не новый выбор модели для эксплуатации по test: полная таблица всех методов сохранена, параметры моделей уже заморожены по validation.",
        table(pd.DataFrame(baseline_comparisons)),
        "## Источники",
        "[Статья](https://publications.hse.ru/pubs/share/direct/861165160.pdf), [архив Антоновой](https://github.com/at-antonova/time_series_prediction/tree/f78c323a461e1f990105546b24d2a548c68f0a6e), [OPSD2018](https://data.open-power-system-data.org/time_series/2018-06-30), [OPSD2020](https://data.open-power-system-data.org/time_series/2020-10-06). Повторный поиск по DOI/названию и проверка связанных репозиториев не обнаружили подтверждённую авторскую реализацию; отсутствие кода в интернете этим не доказано.",
        "Первая попытка поиска до исправления Wishart сохранена в ../reconstruction_before_merge_fix. Она была остановлена до test и повторена после исправления с теми же правилами. GPT Astra проверял алгоритм и протокол в режиме read-only.",
    ]
    (root / "REPORT.md").write_text("\n\n".join(report) + "\n")
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"artifact_checksums.json", "motifs.npz"}
    }
    (root / "artifact_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/reconstruction")
    build(Path(parser.parse_args().output))


if __name__ == "__main__":
    main()
