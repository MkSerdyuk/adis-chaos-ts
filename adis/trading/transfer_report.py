import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .report import formatted_table
from .universe import ROOT


def build_report():
    completed = json.loads((ROOT / "completed.json").read_text())
    metrics = pd.read_csv(ROOT / "metrics.csv")
    coverage = pd.read_csv(ROOT / "coverage.csv")
    statistics = json.loads((ROOT / "bootstrap.json").read_text())
    universe = json.loads((ROOT / "universe.json").read_text())
    primary = next(row for row in statistics if row["primary"])
    if primary["upper"] < 0:
        conclusion = "Средний дневной ES95 метода ниже, чем у причинного контроля: основной блочный интервал целиком ниже нуля."
    elif primary["lower"] > 0:
        conclusion = "Средний дневной ES95 метода выше, чем у причинного контроля: основной блочный интервал целиком выше нуля."
    else:
        conclusion = "Основной блочный интервал разности ES95 включает ноль. Преимущество метода по хвостовому риску не подтверждено; это не доказательство эквивалентности."
    active = int((~coverage.loc[coverage.period == "holdout", "retired_slot"]).sum())
    figures, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    differences = []
    for axis, period, title in zip(
        axes,
        ["historical", "holdout"],
        ["2022–2025: перенос", "2026, январь–октябрь: новый период"],
        strict=True,
    ):
        row = next(item for item in statistics if item["period"] == period and item["block_days"] == 14)
        estimates = row["asset_differences"] + [row["mean_difference"]]
        lower, upper = row["asset_lower"] + [row["lower"]], row["asset_upper"] + [row["upper"]]
        labels = [symbol.removesuffix("USDT") for symbol in row["symbols"]] + ["Среднее пяти слотов"]
        for i, (label, value, lo, hi) in enumerate(zip(labels, estimates, lower, upper, strict=True)):
            axis.plot([lo * 100, hi * 100], [i, i], color="C0" if i < 5 else "C3", linewidth=2)
            axis.scatter(value * 100, i, color="C0" if i < 5 else "C3", s=35)
            differences.append(
                dict(
                    period=period,
                    asset=label,
                    difference=value,
                    lower=lo,
                    upper=hi,
                    interval="99% per asset (Bonferroni family=5)" if i < 5 else "95% aggregate",
                )
            )
        axis.axvline(0, color="grey", linestyle="--", linewidth=1)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.invert_yaxis()
        axis.set(title=title, xlabel="ES95 метода − ES95 контроля, п.п.")
        axis.grid(axis="x", alpha=0.2)
    figures.savefig(ROOT / "comparison.png", dpi=180)
    figures.savefig(ROOT / "comparison.pdf")
    plt.close(figures)
    differences = pd.DataFrame(differences)
    differences.to_csv(ROOT / "risk_differences.csv", index=False)
    comparisons = []
    for period in ["historical", "holdout"]:
        for symbol in completed["symbols"]:
            subset = metrics.loc[(metrics.period == period) & (metrics.symbol == symbol)].set_index(
                "strategy"
            )
            method, control = subset.loc["partial_prediction"], subset.loc["causal_exposure_control"]
            comparisons.append(
                dict(
                    period=period,
                    symbol=symbol,
                    retired_slot=bool(method.retired_slot),
                    inactive_pair=bool(max(method.mean_exposure, control.mean_exposure) < 0.01),
                    method_es95=method.es95,
                    control_es95=control.es95,
                    es_difference=method.es95 - control.es95,
                    method_return=method.total_return,
                    control_return=control.total_return,
                    method_exposure=method.mean_exposure,
                    control_exposure=control.mean_exposure,
                )
            )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(ROOT / "primary_comparison.csv", index=False)
    display = comparison.copy()
    for name in [
        "method_es95",
        "control_es95",
        "method_return",
        "control_return",
        "method_exposure",
        "control_exposure",
    ]:
        display[name] = display[name].map(lambda value: f"{value * 100:.2f}%")
    display.es_difference = display.es_difference.map(lambda value: f"{value * 100:+.3f} п.п.")
    tables = metrics.copy()
    tables["strategy"] = tables.strategy.replace({"partial_prediction": "partial_prediction_tuned"})
    columns = [
        "symbol",
        "strategy",
        "total_return",
        "max_drawdown",
        "daily_volatility",
        "es95",
        "mean_exposure",
        "entries",
    ]
    sections = "".join(
        f'<h2>{title}</h2><div class="scroll">{formatted_table(tables.loc[tables.period == period], columns)}</div>'
        for period, title in [
            ("historical", "Все стратегии: 2022–2025"),
            ("holdout", "Все стратегии: январь–октябрь 2026"),
        ]
    )
    aggregate = pd.DataFrame(
        [
            {key: row[key] for key in ["period", "block_days", "mean_difference", "lower", "upper"]}
            for row in statistics
        ]
    )
    for name in ["mean_difference", "lower", "upper"]:
        aggregate[name] = aggregate[name].map(lambda value: f"{value * 100:+.3f} п.п.")
    ranked = pd.DataFrame(universe["ranked"])[["symbol", "observed_hours", "quote_volume_2019"]]
    report = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Partial multistep prediction: перенос между криптоактивами</title>
<style>body{{max-width:1280px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203040;background:#fafcfe}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dce3ea;text-align:right}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}.panel{{padding:24px;background:white;border:1px solid #dce3ea;border-radius:10px}}img{{width:100%}}pre{{padding:20px;background:#edf2f7;overflow:auto}}a{{color:#145fa1}}</style>
<h1>Partial multistep prediction: перенос между криптоактивами</h1>
<p>Gromov–Baranov, возмущённые траектории с усреднением и rgdbscan. Торговая надстройка на Backtesting.py.</p>
<div class="panel"><strong>{html.escape(conclusion)}</strong><p>Январь–октябрь 2026: средняя разность ES95
{primary["mean_difference"] * 100:+.3f} п.п.; 95% интервал [{primary["lower"] * 100:+.3f}; {primary["upper"] * 100:+.3f}] п.п.
Исторически выбрано пять пар; активных в этом периоде — {active}. EOS — закрытый слот с деньгами.</p></div>
<p><a href="../test/report.html">Исходное исследование BTC и интерактивный бэк-тест</a> ·
<a href="metrics.csv">Все метрики CSV</a> · <a href="primary_comparison.csv">Основное сравнение CSV</a> ·
<a href="../../../research/trading/transfer_protocol.json">Протокол</a> · <a href="frozen.json">Хеши входов и кода</a> ·
<a href="audit.json">Проверка исполнения и бухгалтерии</a></p>
<h2>Как исключён выбор удобного рынка</h2>
<p>Состав определён до расчёта стратегий: пять не-BTC пар USDT с наибольшим оборотом в котируемой валюте за 2019 год.
Требовались торги с начала 2019 года, все 12 месячных архивов и не менее 99% реальных часов. Stablecoins,
фиатные и leveraged-токены исключались по заранее заданному правилу. Список восстановлен по историческим
архивам Binance; сохранённость абсолютно всех прежних тикеров доказать нельзя.
<a href="universe.json">Все кандидаты, исключения, обороты и контрольные суммы</a>.</p>
<p>Параметры перенесены из BTC trial 26 без подбора на новых активах. Для каждого актива мотивы переобучаются
поквартально только на предыдущих 10 000 часах. 2021 год нужен для истории сигналов контрольной стратегии,
а не для нового выбора параметров. Цена нормируется по train, горизонт — 24 часа. Стратегия long/cash,
без плеча, вход на 95% капитала при принятом прогнозе роста более 0,3%; издержки — 15 bps с каждой стороны.
Комиссионная модель одинакова, а фактическая ликвидность и проскальзывание разных активов могут различаться.</p>
<p>Результаты BTC за 2022–2025 уже были известны при постановке расширения. Поэтому перенос на тот же период
показан как дополнительная историческая проверка. Основная проверка — заранее закрытый от исследования
период январь–октябрь 2026; это исторический holdout, а не сбор будущих данных. Оба периода начинают с
отдельных 100 000 USDT. Ни один результат не меняет рынки, параметры, пороги или даты.</p>
<h2>Основное сравнение</h2><div class="scroll">{display.to_html(index=False, border=0)}</div>
<p>ES95 — средняя потеря в худших ceil(5% × n) днях. Контроль ежедневно поддерживает долю актива по среднему
предыдущих 90 targets метода. Показываются фактические доли: они могут различаться. Средняя разность по
пяти слотам — характеристика качества переноса, а не риск инвестиционного портфеля. Нулевая активность
не считается успехом распознавания риска.</p><img src="comparison.png" alt="Разности хвостового риска метода и контроля на всех выбранных рынках">
<p>На рисунке интервалы отдельных активов — 99% (поправка Bonferroni на пять сравнений), общего среднего — 95%.
Это приближённые bootstrap-интервалы. 10 000 повторов используют одинаковые календарные блоки для всех
активов и обеих стратегий, сохраняя их корреляцию. Основной блок — 14 дней, 7 и 28 дней проверяют чувствительность.
Криптоактивы не считаются независимыми наблюдениями; лучший рынок не выбирается для заявления о превосходстве.</p>
{aggregate.to_html(index=False, border=0)}
<h2>EOS и прекращение торгов</h2><p>EOS вошёл в список по обороту 2019 года и остался в нём.
<a href="https://www.binance.com/en/support/announcement/detail/1e89a9ca957c4b0ca7502e60b993e201">Уведомление биржи</a>
датировано 14 мая 2025; поскольку часовой пояс публикации не подписан, правило доступности добавляет 48 часов
после начала этой даты UTC. Все стратегии переводят target в ноль после закрытия бара 15 мая 23:00,
исполнение — 16 мая 01:00, до прекращения торгов 26 мая 03:00 UTC. После этого новые мотивы не строятся,
доходность денег равна нулю. EOS не заменён на следующий по обороту актив и не сшит с Vaulta/A.
Его вклад в 2026 равен нулю, знаменатель среднего остаётся пять; это не пятый активный рынок с успешным прогнозом.</p>
{sections}
<h2>Охват прогнозов</h2>{coverage.to_html(index=False, border=0)}
<h2>Рейтинг до теста</h2>{ranked.to_html(index=False, border=0, float_format=lambda value: f"{value:,.0f}")}
<h2>Воспроизведение</h2><p>Готовые universe и selection уже сохранены. Восстановление данных и расчётов:</p>
<pre>uv sync --locked --extra trading
uv run --extra trading python -m adis.trading.transfer_data
uv run --extra trading python -m adis.trading.transfer
uv run --extra trading python -m adis.trading.audit --transfer
uv run --extra trading python -m adis.trading.transfer_report</pre>
<p>Проверяется исполнение только на реальных Open, комиссии, PnL и конечный капитал; дневные доходности,
ES и просадки пересчитаны независимо. Даты обучения и код зафиксированы хешами. Полный перебор сохранён,
включая отрицательные результаты. Новая оптимизация после просмотра этого отчёта потребует другого внешнего теста.</p>
</html>"""
    (ROOT / "report.html").write_text(report)
    summary = dict(
        conclusion=conclusion,
        primary=primary,
        selected_assets=5,
        active_holdout_assets=active,
        comparisons=comparisons,
        backtests=completed["backtests"],
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(ROOT / "report.html")


if __name__ == "__main__":
    build_report()
