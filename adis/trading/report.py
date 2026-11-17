import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import ROOT

LABELS = {
    "partial_prediction_tuned": "Partial multistep prediction · TPE",
    "partial_prediction_paper": "Partial multistep prediction · параметры статьи",
    "buy_hold": "Buy & hold",
    "sma_50_200_daily": "SMA 50/200",
    "momentum_90_daily": "Momentum 90",
    "donchian_20_10_daily": "Donchian 20/10",
    "rsi_14_daily": "RSI 14",
    "vol_target_20pct_30daily": "Vol-target 20%",
    "random_forest_24h_return": "Random Forest",
    "partial_prediction_cm": "Partial multistep prediction без growth gate",
    "forced_nearest_avg": "Ближайшие мотивы + avg",
    "causal_exposure_control": "Контроль по прошлым 90 дням",
    "retrospective_exposure_control": "Ретроспективная постоянная доля",
    "cash": "Деньги",
}


def formatted_table(frame, columns):
    table = frame[columns].copy()
    if "strategy" in table:
        table["strategy"] = table.strategy.map(LABELS)
    if "comparator" in table:
        table["comparator"] = table.comparator.map(LABELS)
    percentages = [
        "total_return",
        "cagr",
        "max_drawdown",
        "daily_volatility",
        "es95",
        "mean_exposure",
        "worst_day",
        "downside_deviation",
        "difference",
        "lower",
        "upper",
    ]
    for column in percentages:
        if column in table:
            suffix = " п.п." if column in {"difference", "lower", "upper"} else "%"
            table[column] = table[column].map(lambda value, suffix=suffix: f"{100 * value:.2f}{suffix}")
    return table.to_html(index=False, border=0, classes="metrics", escape=True)


def build_report():
    folder = ROOT / "test"
    completed = json.loads((folder / "completed.json").read_text())
    selected = json.loads((ROOT / "selected.json").read_text())
    development = json.loads((ROOT / "development_history.json").read_text())
    configurations = []
    for filename in ["v2_engineering_pilot_trials.json", "v3_quarter_activity_trials.json"]:
        for trial in json.loads((ROOT / filename).read_text()):
            if trial["state"] == "COMPLETE":
                configurations.append(trial["attributes"]["parameters"])
    configurations.extend(
        json.loads(path.read_text())["parameters"] for path in (ROOT / "validation").glob("*/result.json")
    )
    unique_configurations = len({json.dumps(value, sort_keys=True) for value in configurations})
    prior_trials = sum(study["complete"] for study in development["studies"])
    metrics = pd.read_csv(folder / "metrics.csv")
    yearly = pd.read_csv(folder / "yearly_metrics.csv")
    intervals = pd.read_csv(folder / "bootstrap.csv")
    sensitivity = pd.read_csv(folder / "sensitivity.csv")
    seeds = pd.read_csv(folder / "seeds.csv")
    coverage = pd.read_csv(folder / "forecast_coverage.csv")
    conditional = pd.read_csv(folder / "conditional_risk.csv")
    equity = pd.read_csv(folder / "equities.csv.gz", index_col=0, parse_dates=True)
    daily = equity.resample("1D").last()
    primary = intervals.loc[intervals.primary].iloc[0]
    tuned = metrics.set_index("strategy").loc["partial_prediction_tuned"]
    control = metrics.set_index("strategy").loc["causal_exposure_control"]
    without_costs = sensitivity.loc[
        (sensitivity.strategy == "partial_prediction_tuned")
        & (sensitivity.costs_bps == 0)
        & (sensitivity.delay_hours == 1)
    ].iloc[0]
    ordinary = metrics.loc[
        metrics.strategy.isin(
            [
                "buy_hold",
                "sma_50_200_daily",
                "momentum_90_daily",
                "donchian_20_10_daily",
                "rsi_14_daily",
                "vol_target_20pct_30daily",
                "random_forest_24h_return",
            ]
        )
    ]
    improvements = ordinary.loc[
        (ordinary.max_drawdown > tuned.max_drawdown)
        & (ordinary.daily_volatility > tuned.daily_volatility)
        & (ordinary.es95 > tuned.es95)
    ]
    conclusion = (
        f"У Partial multistep prediction ниже одновременно просадка, волатильность и ES95, чем у {len(improvements)} "
        f"из {len(ordinary)} обычных стратегий в этом тесте."
    )
    if primary.upper < 0:
        inference = "Парный блочный интервал основной разности ES95 целиком ниже нуля: результат благоприятен для Partial multistep prediction против заданного причинного контроля."
    elif primary.lower > 0:
        inference = "Парный блочный интервал основной разности ES95 целиком выше нуля: Partial multistep prediction имеет больший хвостовой риск, чем заданный причинный контроль."
    else:
        inference = "Парный блочный интервал основной разности ES95 включает ноль: устойчивое преимущество перед заданным причинным контролем не установлено."
    eligibility = (
        "Выбранная настройка выполнила ограничения активности на validation."
        if selected["eligible"]
        else "Ни одна настройка не выполнила все ограничения активности на validation. Показан заранее предусмотренный выбор по наименьшему нарушению ограничений; его нельзя считать прошедшей отбор стратегией."
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    shown = [
        "partial_prediction_tuned",
        "buy_hold",
        "sma_50_200_daily",
        "vol_target_20pct_30daily",
        "random_forest_24h_return",
        "causal_exposure_control",
    ]
    for name in shown:
        axes[0, 0].plot(daily.index, daily[name] / 100000, label=LABELS[name], linewidth=1.4)
        drawdown = 1 - equity[name] / equity[name].cummax().clip(lower=100000)
        axes[0, 1].plot(
            drawdown.resample("1D").max().index,
            100 * drawdown.resample("1D").max(),
            label=LABELS[name],
            linewidth=1,
        )
    axes[0, 0].set(title="Капитал после издержек", ylabel="Начальный капитал = 1", yscale="log")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Просадка по часовому equity", ylabel="%")
    comparison = metrics.set_index("strategy")
    for name, color in [
        ("partial_prediction_tuned", "C0"),
        ("causal_exposure_control", "C5"),
        ("buy_hold", "C1"),
    ]:
        row = comparison.loc[name]
        axes[1, 0].scatter(
            100 * row.mean_exposure,
            100 * row.es95,
            s=50,
            color=color,
            label=LABELS[name],
        )
    axes[1, 0].set(
        title="Основное сравнение: риск и доля BTC",
        xlabel="Средняя экспозиция, %",
        ylabel="Дневной ES95, %",
        xlim=(0, 100),
        ylim=(0, 7),
    )
    axes[1, 0].legend(fontsize=8, loc="upper left")
    subset = yearly[yearly.strategy.isin(["partial_prediction_tuned", "causal_exposure_control", "buy_hold"])]
    pivot = subset.pivot(index="year", columns="strategy", values="es95")
    for i, name in enumerate(pivot.columns):
        axes[1, 1].bar(
            np.arange(len(pivot)) + (i - 1) * 0.25,
            pivot[name] * 100,
            width=0.25,
            label=LABELS[name],
            color={"partial_prediction_tuned": "C0", "causal_exposure_control": "C5", "buy_hold": "C1"}[name],
        )
    axes[1, 1].set_xticks(np.arange(len(pivot)), pivot.index)
    axes[1, 1].set(title="ES95 по календарным годам", ylabel="%")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(folder / "comparison.png", dpi=180)
    figure.savefig(folder / "comparison.pdf")
    plt.close(figure)
    columns = [
        "strategy",
        "total_return",
        "cagr",
        "max_drawdown",
        "daily_volatility",
        "es95",
        "mean_exposure",
        "entries",
    ]
    sources = json.loads(Path("research/trading/sources.json").read_text())
    links = "".join(
        f'<li><a href="{html.escape(source["url"])}">{html.escape(source["title"])}</a>: '
        f"{html.escape(source['use'])}</li>"
        for source in sources
    )
    parameters = html.escape(json.dumps(selected["parameters"], indent=2))
    report = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Partial multistep prediction: торговля BTC/USDT и риск</title>
<style>body{{max-width:1280px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203040;background:#fafcfe}}
h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #dce3ea;text-align:right}}td:first-child,th:first-child{{text-align:left}}pre{{background:#edf2f7;padding:20px;overflow:auto}}img{{width:100%}}.lead{{font-size:20px}}.panel{{background:white;border:1px solid #dce3ea;border-radius:10px;padding:22px;margin:24px 0}}a{{color:#145fa1}}.scroll{{overflow:auto}}</style>
<h1>Partial multistep prediction в торговле BTC/USDT: проверка риска</h1>
<p>Метод Gromov–Baranov из статьи «Prediction after a Horizon of Predictability: Nonpredictable Points and
Partial Multistep Prediction for Chaotic Time Series». Используется вариант с возмущёнными траекториями,
усреднением конечных значений и правилом отказа rgdbscan. Торговые правила добавлены в этом исследовании.</p>
<p class="lead">{html.escape(inference)}</p><p>{html.escape(conclusion)} {html.escape(eligibility)}</p>
<div class="panel"><strong>Основной результат:</strong> разность дневного ES95 Partial multistep prediction − причинный контроль:
{primary.difference * 100:.3f} процентного пункта; 95% блочный интервал [{primary.lower * 100:.3f}; {primary.upper * 100:.3f}].
Средняя фактическая доля BTC: {tuned.mean_exposure * 100:.2f}% у Partial multistep prediction и {control.mean_exposure * 100:.2f}% у контроля.
Это сравнение с причинно сглаженной долей, а не точное равенство риск-бюджетов.</div>
<p>Чистая доходность Partial multistep prediction за четыре года: {tuned.total_return * 100:.2f}%; у причинного контроля:
{control.total_return * 100:.2f}%. При нулевых издержках и той же задержке результат Partial multistep prediction:
{without_costs.total_return * 100:.2f}%. Поэтому весь результат нельзя объяснить одними комиссиями.</p>
<p><a href="partial_prediction_backtest.html">Открыть интерактивный бэк-тест: капитал, сделки и просадки</a> ·
<a href="metrics.csv">Все метрики CSV</a> · <a href="../selected.json">Выбранные параметры</a> ·
<a href="../../../research/trading/protocol.json">Протокол</a></p>
<h2>Данные и исполнение</h2><p>Официальные часовые свечи Binance BTCUSDT spot, 2019–2025. Архивы проверены SHA256.
63 отсутствующих или нулевых часа отмечены отдельно. Синтетические свечи используются только для причинного
заполнения сетки модели; на них сделки не исполняются. Начальный капитал 100 000 USDT, без плеча и шортов,
доходность денег равна нулю. Решение после закрытия бара 23:00 UTC, исполнение на следующем реальном открытии
после ещё одного законченного часа: обычно в 01:00 UTC. Издержки 15 bps в каждую сторону:
10 bps комиссии плюс 5 bps дополнительной пропорциональной платы за проскальзывание.</p>
<h2>Подбор и проверка</h2><p>Финальный Optuna TPE: 32 оценки, четыре последовательных квартала 2021 года.
Критерий: средний квартальный ES95 плюс половина его стандартного отклонения. Ограничения: средняя экспозиция
не ниже 10% и минимум 12 входов за validation-год. Затем параметры заморожены;
тест 2022–2025 выполняется с квартальным переобучением на последних 10 000 часах. Предыдущие наблюдения теста
становятся обучающей историей только после наступления их времени. В гиперпараметры test не передаётся.</p>
<p>Протокол разрабатывался на validation: до финального поиска выполнено ещё {prior_trials} завершённых оценок.
Всего сохранено {len(configurations)} завершённых оценок и {unique_configurations} уникальных конфигураций,
часть повторов использовала кэш. Первый инженерный запуск остановлен для усиления проверки кэшей.
В следующем запуске было требование 2% экспозиции в каждом квартале; первые 16 вариантов его не выполнили.
Его сняли до просмотра test, сохранив общие ограничения активности. Длительный отказ от торговли допустим
для этого метода. Отсутствие входов в Q1 2021 связано с выходом цены за исторический диапазон мотивов,
а само по себе не доказывает распознавание повышенного риска. Все предыдущие trials сохранены.</p>
<p>Разрешённый пользователем поиск меняет L/K, r/μ Wishart и параметры прогноза. Все комбинации промежутков
выбранного L/K сохраняются, Wishart включён. Прогноз всегда на 24 часа. Long на 95% доступного капитала при
принятом прогнозе роста более 0,3%, иначе cash. Неизменный сигнал не вызывает повторного входа; отдельно
указанные exposure controls и vol-target ребалансируются ежедневно. Правило торговли — новая надстройка,
не торговая стратегия из исходной статьи. Её опубликованная конфигурация оставлена отдельным контролем.</p>
<pre>{parameters}</pre>
<h2>Общие результаты после издержек</h2><div class="scroll">{formatted_table(metrics, columns)}</div>
<p>daily_volatility — стандартное отклонение дневных доходностей × √365. ES95 — средняя потеря в худших
ceil(5% × n) днях, ограниченная снизу нулём. max_drawdown использует часовую переоценку по Close; внутрисвечные
просадки не измерены. Экспозиция фактическая, после изменения стоимости позиций. Доходность публикуется
вместе с риском: уменьшение риска не означает прибыльности.</p><img src="comparison.png" alt="Капитал, просадки и риск">
<h2>По годам</h2><div class="scroll">{formatted_table(yearly, ["year"] + columns)}</div>
<h2>Отказ от прогноза и движения рынка</h2>{coverage.to_html(index=False, border=0)}
{conditional.to_html(index=False, border=0, float_format=lambda value: f"{value:.5f}")}
<p>Эти условные группы — диагностика, а не доходность стратегии. Последний 24-часовой target вне диапазона данных
не оценивается. «cm» отключает growth gate, но сохраняет отказ при отсутствии соседей; forced nearest использует
другой агрегатор avg. Потоки шума cm и rgdbscan расходуются по-разному из-за дополнительных пробных шагов,
поэтому их сравнение не изолирует единственное булево решение при идентичных случайных траекториях.</p>
<h2>Неопределённость</h2>{formatted_table(intervals, ["comparator", "primary", "block_days", "difference", "lower", "upper"])}
<p>Парный circular moving-block bootstrap: 2000 повторов, основной блок 14 дней, чувствительность 7/28 дней.
Одна основная гипотеза — сравнение с причинной долей по прошлым 90 targets. Остальные интервалы описательные.
Ретроспективная постоянная доля вычислена по всему тесту: это не исполнимая заранее стратегия, а диагностика;
её bootstrap не учитывает неопределённость самой выбранной доли. Четыре года одного рынка не доказывают
снижение риска на любых будущих режимах.</p>
<h2>Другие seed</h2>{formatted_table(seeds, ["seed"] + columns[1:])}
<h2>Издержки и задержка</h2><div class="scroll">{formatted_table(sensitivity, ["strategy", "costs_bps", "delay_hours", "total_return", "max_drawdown", "es95", "mean_exposure"])}</div>
<h2>Воспроизводимость</h2><p>Готовая платформа — Backtesting.py FractionalBacktest. В репозитории есть ledger прогнозов,
границы обучения, хеши моделей/данных/кода, сделки, заявки и почасовой капитал. Тесты проверяют причинность,
независимость RNG от будущих origins, точный пример комиссий и исполнения, годовую базу equity и ребалансировку.
<a href="../model_manifest.json">Манифест моделей</a> · <a href="../data_manifest.json">Манифест данных</a> ·
<a href="frozen.json">Замороженный вход теста</a>.</p>
<pre>uv sync --extra trading
uv run --extra trading python -m adis.trading.data
uv run --extra trading python -m adis.trading.experiment tune --trials 32
uv run --extra trading python -m adis.trading.evaluation
uv run --extra trading python -m adis.trading.audit
uv run --extra trading python -m adis.trading.report</pre>
<h2>Источники и границы сравнения</h2><ul>{links}</ul>
<p>Чужие опубликованные доходности с иных периодов не перенесены в таблицу. Random Forest и технические стратегии
реализованы как общие бенчмарки и исполнены на том же календаре и с теми же издержками. Оптимизация Partial multistep prediction и
фиксированные стандартные настройки бенчмарков не позволяют объявлять превосходство над всеми их возможными настройками.</p>
</html>"""
    (folder / "report.html").write_text(report)
    summary = dict(
        conclusion=conclusion,
        primary_inference=inference,
        validation_eligible=selected["eligible"],
        primary_bootstrap=primary.to_dict(),
        tuned_metrics=tuned.to_dict(),
        causal_control_metrics=control.to_dict(),
        completed=completed,
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(folder / "report.html")


if __name__ == "__main__":
    build_report()
