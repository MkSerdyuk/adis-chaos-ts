import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .equity_data import ROOT

LABELS = {
    "partial_prediction": "Partial multistep prediction",
    "paper_control": "Контроль L4 K10 из статьи",
    "causal_control": "Доля по предыдущим 13 неделям",
    "cash": "Деньги",
    "buy_hold": "Buy & hold",
    "sma50_200": "SMA 50/200",
    "momentum90": "Momentum 90",
    "donchian20_10": "Donchian 20/10",
    "rsi14": "RSI 14",
    "vol_target20pct_30": "Vol-target 20%",
    "random_forest5": "Random Forest",
}


def table(frame):
    frame = frame.copy()
    if "strategy" in frame:
        frame["strategy"] = frame.strategy.map(LABELS)
    for column in [
        "total_return",
        "max_drawdown",
        "volatility",
        "es95",
        "mean_exposure",
        "method_return",
        "control_return",
        "method_es95",
        "control_es95",
        "method_exposure",
        "control_exposure",
    ]:
        if column in frame:
            frame[column] = frame[column].map(lambda value: f"{value * 100:.2f}%")
    for column in ["difference", "lower", "upper"]:
        if column in frame:
            frame[column] = frame[column].map(lambda value: f"{value * 100:+.3f} п.п.")
    return (
        '<div class="scroll">'
        + frame.to_html(index=False, border=0, float_format=lambda value: f"{value:,.2f}")
        + "</div>"
    )


def build_report():
    completed = json.loads((ROOT / "completed.json").read_text())
    audit = json.loads((ROOT / "artifact_audit.json").read_text())
    frozen = completed["frozen"]
    metrics = pd.read_csv(ROOT / "metrics.csv")
    sensitivity = pd.read_csv(ROOT / "sensitivity.csv")
    seeds = pd.read_csv(ROOT / "seeds.csv")
    coverage = pd.read_csv(ROOT / "coverage.csv")
    statistics = json.loads((ROOT / "bootstrap.json").read_text())
    primary = next(row for row in statistics if row["primary"])
    if primary["upper"] < 0:
        conclusion = "Средняя по трём классам разность ES95 благоприятна методу: основной блочный интервал целиком ниже нуля. Это результат данной дневной адаптации на фиксированной панели."
    elif primary["lower"] > 0:
        conclusion = "Метод имеет больший средний хвостовой риск, чем причинный контроль: основной блочный интервал целиком выше нуля."
    else:
        conclusion = "Устойчивое преимущество по хвостовому риску не подтверждено: основной интервал разности ES95 включает ноль. Это не доказывает эквивалентность стратегий."
    comparisons = []
    for period in ["historical", "holdout"]:
        interval = next(row for row in statistics if row["period"] == period and row["block"] == 10)
        for i, symbol in enumerate(frozen["symbols"]):
            subset = metrics.loc[(metrics.period == period) & (metrics.symbol == symbol)].set_index(
                "strategy"
            )
            method, control = subset.loc["partial_prediction"], subset.loc["causal_control"]
            comparisons.append(
                dict(
                    period=period,
                    symbol=symbol,
                    method_return=method.total_return,
                    control_return=control.total_return,
                    method_es95=method.es95,
                    control_es95=control.es95,
                    difference=method.es95 - control.es95,
                    lower=interval["asset_lower"][i],
                    upper=interval["asset_upper"][i],
                    method_exposure=method.mean_exposure,
                    control_exposure=control.mean_exposure,
                    method_entries=int(method.entries),
                    inactive_pair=bool(max(method.mean_exposure, control.mean_exposure) < 0.01),
                )
            )
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(ROOT / "primary_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    figure.suptitle("Partial multistep prediction: акции, облигации и золото")
    figure.supxlabel(
        "Интервалы инструментов: 99,286% (Bonferroni, 7 сравнений); среднего по классам: 95%. Отрицательная разность благоприятна методу.",
        fontsize=9,
    )
    for axis, period, title in zip(
        axes,
        ["historical", "holdout"],
        ["2022–2025: дополнительный период", "Январь–октябрь 2026: основная оценка"],
        strict=True,
    ):
        row = next(value for value in statistics if value["period"] == period and value["block"] == 10)
        labels = row["symbols"] + ["Среднее по классам"]
        for i, (value, lo, hi) in enumerate(
            zip(
                row["asset_differences"] + [row["difference"]],
                row["asset_lower"] + [row["lower"]],
                row["asset_upper"] + [row["upper"]],
                strict=True,
            )
        ):
            color = "C0" if i < 7 else "C3"
            axis.plot([lo * 100, hi * 100], [i, i], color=color, linewidth=2)
            axis.scatter(value * 100, i, color=color, s=35)
        axis.axvline(0, color="grey", linestyle="--", linewidth=1)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.invert_yaxis()
        axis.set(title=title, xlabel="ES95 метода − ES95 контроля, п.п.")
        axis.grid(axis="x", alpha=0.2)
    figure.savefig(ROOT / "risk_comparison.png", dpi=180)
    figure.savefig(ROOT / "risk_comparison.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(4, 2, figsize=(13, 11), layout="constrained")
    for axis, symbol in zip(axes.ravel(), frozen["symbols"], strict=False):
        for name, color in [("partial_prediction", "C0"), ("causal_control", "C1"), ("buy_hold", "C2")]:
            curve = pd.read_csv(
                ROOT / symbol / "holdout" / name / "equity.csv.gz", index_col=0, parse_dates=True
            )
            axis.plot(curve.index, curve.equity / 100000, label=LABELS[name], color=color, linewidth=1.3)
        axis.set(title=symbol, ylabel="Начальный капитал = 1")
        axis.grid(alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].legend(handles, labels, loc="center", frameon=False)
    figure.suptitle("Капитал с дивидендами и издержками: январь–октябрь 2026")
    figure.savefig(ROOT / "equities.png", dpi=180)
    figure.savefig(ROOT / "equities.pdf")
    plt.close(figure)
    aggregate = pd.DataFrame(
        [
            {key: row[key] for key in ["period", "sessions", "block", "difference", "lower", "upper"]}
            for row in statistics
        ]
    )
    seed_risks, sensitivity_risks = [], []
    for (period, seed), group in seeds.groupby(["period", "seed"]):
        pivot = group.pivot(index="symbol", columns="strategy", values="es95").reindex(frozen["symbols"])
        seed_risks.append(
            dict(
                period=period,
                seed=seed,
                difference=float(
                    (pivot.partial_prediction - pivot.causal_control).to_numpy()
                    @ np.array(primary["weights"])
                ),
            )
        )
    for (period, cost, delay, dividend_delay), group in sensitivity.groupby(
        ["period", "costs_bps", "execution_delay_sessions", "dividend_delay_days"]
    ):
        pivot = group.pivot(index="symbol", columns="strategy", values="es95").reindex(frozen["symbols"])
        sensitivity_risks.append(
            dict(
                period=period,
                costs_bps=cost,
                extra_delay_sessions=delay,
                dividend_delay_days=dividend_delay,
                difference=float(
                    (pivot.partial_prediction - pivot.causal_control).to_numpy()
                    @ np.array(primary["weights"])
                ),
            )
        )
    sources = json.loads(Path("research/trading/noncrypto/sources.json").read_text())
    links = "".join(
        f'<li><a href="{html.escape(row["url"])}">{html.escape(row["title"])}</a>: {html.escape(row["use"])}</li>'
        for row in sources
    )
    columns = [
        "symbol",
        "strategy",
        "total_return",
        "max_drawdown",
        "volatility",
        "es95",
        "mean_exposure",
        "entries",
        "dividends",
    ]
    settings = html.escape(
        json.dumps(
            dict(
                transferred=frozen["parameters"],
                paper_control=frozen["paper_parameters"],
                training_sessions=2520,
                horizon_sessions=5,
                control_weeks=13,
            ),
            indent=2,
        )
    )
    report = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Partial multistep prediction вне криптовалют</title>
<style>body{{max-width:1300px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203040;background:#fafcfe}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dce3ea;text-align:right}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}.panel{{background:white;border:1px solid #dce3ea;border-radius:10px;padding:24px}}img{{width:100%}}pre{{padding:20px;background:#edf2f7;overflow:auto}}a{{color:#145fa1}}</style>
<h1>Partial multistep prediction вне криптовалют</h1><p>Метод Gromov–Baranov: возмущённые траектории, усреднение конечных значений и rgdbscan. Дневная торговая адаптация на Backtesting.py.</p>
<div class="panel"><strong>{html.escape(conclusion)}</strong><p>Январь–октябрь 2026: разность ES95
{primary["difference"] * 100:+.3f} п.п.; 95% интервал [{primary["lower"] * 100:+.3f}; {primary["upper"] * 100:+.3f}] п.п.
Основная оценка использует {primary["sessions"]} торговых сессий и равные веса трёх классов активов.</p></div>
<p><a href="stock_backtest.html">Интерактивный пример SPY</a> · <a href="metrics.csv">Все метрики CSV</a> ·
<a href="primary_comparison.csv">Основное сравнение CSV</a> · <a href="../../../research/trading/noncrypto/protocol.json">Протокол</a> ·
<a href="frozen.json">Хеши входов и кода</a> · <a href="artifact_audit.json">Проверка сохранённых расчётов</a></p>
<h2>Что проверялось</h2><p>Панель зафиксирована до расчёта: AAPL, MSFT и XOM — отдельные акции; SPY и QQQ — фондовые ETF;
TLT — длинные государственные облигации США; GLD — золото. Это выбранные известные сохранившиеся инструменты,
а не выборка всех исторически доступных акций без survivorship bias. Нельзя переносить вывод автоматически на весь фондовый рынок.</p>
<p>Предыдущие криптоэксперименты уже были известны. 2022–2025 здесь — дополнительный исторический период,
январь–октябрь 2026 — ранее не рассчитанный для этих инструментов holdout, но общий макроэкономический период
с криптоэкспериментом. Это исследовательское расширение, не полностью независимое проспективное подтверждение.
Рынки не выбирались по полученным доходностям, новый TPE на этой панели не проводился.</p>
<h2>Основное сравнение с причинным контролем</h2>{table(comparisons.loc[comparisons.period == "holdout"])}
<p>Контроль раз в неделю поддерживает долю актива, равную среднему предыдущих 13 недельных targets метода.
Фактическая экспозиция у стратегий может различаться — обе доли показаны. Нулевое участие в рынке само по себе
не является доказательством распознавания риска. Основной эффект — средняя разность ES по классам, не риск инвестиционного портфеля.</p>
<img src="risk_comparison.png" alt="Хвостовой риск метода относительно контроля на семи инструментах">
<p>Каждый из пяти фондовых инструментов имеет вес 1/15, TLT и GLD — по 1/3. Bootstrap: 10 000 повторов,
одни и те же календарные блоки у всех инструментов и обеих стратегий; основной блок — 10 сессий, дополнительные — 5/20.
Для отдельных инструментов используются приблизительные интервалы с поправкой Bonferroni на семь сравнений.
Лучшая строка не выбирается как новый рынок для торговли. Остальные сравнения описательные.</p>{table(aggregate)}
<img src="equities.png" alt="Капитал метода, причинного контроля и buy-and-hold на всех инструментах">
<h2>Изменение временного масштаба</h2><p>Это не буквальный перенос всех настроек часовой криптостратегии.
До просмотра результатов заданы 2520 предыдущих торговых сессий для обучения, прогноз на 5 сессий,
решение после последнего закрытия каждой торговой недели и исполнение на следующем реальном Open.
Переобучение поквартальное; текущий квартал не входит в его train. Недельное расписание учитывает праздники и короткие недели.
Параметры мотивов, Wishart, радиусов и шума перенесены из BTC trial 26. Контроль L4 K10 использует
геометрию статьи и ранее зафиксированные параметры её реконструкции; торговая политика в самой статье не исследовалась.</p>
<pre>{settings}</pre><p>Модель прогнозирует split-adjusted цену Close, не total return. Нормировка — min–max по train,
без обрезки будущих значений; будущие дивиденды модель не получает. Принятый прогноз роста более 0,3% даёт
long на 95% капитала, иначе cash; без плеча и коротких позиций. Начальный капитал каждого периода — 100 000 USD,
проценты на деньги равны нулю. Издержки — 15 bps с каждой стороны: одинаковая заданная модель, не измеренный spread каждого инструмента.</p>
<h2>Данные, сплиты и дивиденды</h2><p>Снимки Yahoo Finance содержат 5503 дневные сессии с 2005 года по 16 ноября 2026.
Последние ноябрьские наблюдения не входят в оценку до 1 ноября. Все даты сверены с календарём биржи.
Сырые ответы сохранены с контрольными суммами; программный endpoint не имеет гарантии стабильного официального API.
На исполнение идут цены с корректировкой только на сплиты; dividend-adjusted Adj Close не подставляется вместо Open.</p>
<p>Используются дробные доли акции в постоянных split-adjusted единицах. При пропорциональных комиссиях смена единиц
сохраняет стоимость позиции и дивиденды. Проекция в raw price/quantity проверяет внутреннюю эквивалентность,
но не является сверкой с независимым архивом биржевых raw quotes. Проверки искусственного будущего сплита
сохраняют прогнозы, targets, капитал и процентные комиссии в пределах точности дробного исполнения.</p>
<p>Дивиденды начисляются отдельно держателю до Open ex-date: продавец на этом открытии получает выплату,
покупатель — нет. Основная версия предполагает доступность денег в ex-date, поскольку источник не задаёт все payment dates.
Дополнительная версия задерживает зачисление на 30 календарных дней: receivable входит в equity, но не доступен
для покупки и сохраняется после продажи. Это проверка допущения, а не реконструкция реальных дат выплаты.
Остаток cash и receivables сохранён отдельно. Итоговый капитал включает dividend ledger;
в интерактивном графике PnL отдельных сделок отражает только цены и комиссии, equity включает дивиденды.</p>
<p><a href="corporate_action_checks.json">Выборочные сверки с эмитентами</a> подтверждают три сплита Apple и выплаты
до и после них. В проверенных выплатах TLT Yahoo округляет сумму до 0,001 USD на акцию, тогда как эмитент даёт
больше знаков. Эта погрешность сохранена и описана; точные суммы не подменяются выборочно.
Все корпоративные действия независимо не сертифицированы.</p>
<h2>Все стратегии: 2022–2025</h2>{table(metrics.loc[metrics.period == "historical", columns])}
<h2>Все стратегии: январь–октябрь 2026</h2>{table(metrics.loc[metrics.period == "holdout", columns])}
<p>ES95 — средняя потеря в худших ceil(5% × n) сессиях, ограниченная снизу нулём. Выходные и праздники не добавляются
как нулевые доходности. Волатильность годовая, множитель √252; просадка по закрытиям, без внутридневных минимумов.
Поле cagr в CSV — годовая экстраполяция на условные 252 сессии. Доходность приведена после издержек и с дивидендами,
без налогов и процентов на cash. Технические стратегии и Random Forest используют фиксированные настройки;
их возможные лучшие гиперпараметры здесь не ищутся.</p>
<h2>Охват прогнозов</h2>{coverage.to_html(index=False, border=0, float_format=lambda value: f"{value:.3f}")}
<h2>Проверки чувствительности</h2><p>Разности ES95 ниже — точечные оценки, не новые независимые проверки гипотезы.
Параметры и основной результат не выбираются по лучшему seed или стоимости.</p>{table(pd.DataFrame(sensitivity_risks))}{table(pd.DataFrame(seed_risks))}
<p><a href="sensitivity.csv">Все метрики чувствительности</a> · <a href="seeds.csv">Seed-проверки</a> · <a href="yearly_metrics.csv">Метрики по годам</a></p>
<h2>Воспроизведение и проверка</h2><p>Выполнены {completed["main_backtests"]} основных бэк-теста,
{completed["sensitivity_backtests"]} проверок условий и {completed["seed_backtests"]} seed-проверок.
Все {audit["backtests_verified"]} расчётов повторно проверены по сохранённым файлам: Open исполнения,
комиссии, price PnL, право на дивиденд, доступные деньги, receivables, дневные доходности и риск.
Даты обучения и хеши данных проверены для {audit["forecast_rows_verified"]} прогнозных origins.
Записи частичных закрытий не являются независимыми статистическими наблюдениями.</p>
<pre>bash research/trading/noncrypto/reproduce.sh</pre>
<p>Команда восстанавливает точный сохранённый snapshot, затем повторяет расчёт и проверку. Кэши прогнозов и
metadata сохраняются; численные модели можно восстановить. Изменение данных или численного кода после freeze
останавливает прежний эксперимент. Все отрицательные и неопределённые результаты остаются в отчёте.</p>
<h2>Источники</h2><ul>{links}</ul></html>"""
    (ROOT / "report.html").write_text(report)
    summary = dict(
        conclusion=conclusion,
        primary=primary,
        comparisons=comparisons.to_dict("records"),
        sensitivity=sensitivity_risks,
        seeds=seed_risks,
        completed={key: value for key, value in completed.items() if key != "frozen"},
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(ROOT / "report.html")


if __name__ == "__main__":
    build_report()
