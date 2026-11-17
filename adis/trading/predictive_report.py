import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .predictive_comparison import CANDIDATES, ROOT

LABELS = {
    "partial_prediction": "Partial multistep prediction",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ridge": "Ridge",
    "svr": "SVR",
    "sma50_200": "SMA 50/200",
    "momentum90": "Momentum 90",
    "buy_hold": "Buy-and-hold",
    "causal_control": "Причинный контроль",
    "paper_control": "Геометрия статьи",
}
PERCENT_COLUMNS = ["total_return", "es95", "max_drawdown", "volatility", "mean_exposure"]


def table(frame):
    formatted = frame.copy()
    formatted["strategy"] = formatted.strategy.map(LABELS)
    for name in PERCENT_COLUMNS:
        formatted[name] = formatted[name].map(lambda value: f"{value * 100:.2f}%")
    formatted = formatted.rename(
        columns={
            "strategy": "Метод",
            "symbol": "Инструмент",
            "total_return": "Доходность",
            "es95": "ES95 за день",
            "max_drawdown": "Макс. просадка",
            "volatility": "Волатильность / год",
            "mean_exposure": "Средняя доля в рынке",
        }
    )
    return '<div class="scroll">' + formatted.to_html(index=False, border=0) + "</div>"


def build_report():
    metrics = pd.read_csv(ROOT / "portfolio_metrics.csv")
    instruments = pd.read_csv(ROOT / "instrument_metrics.csv")
    eligibility = pd.read_csv(ROOT / "eligibility.csv")
    comparisons = pd.read_csv(ROOT / "comparisons.csv")
    audit = json.loads((ROOT / "artifact_audit.json").read_text())
    names = ["partial_prediction", *CANDIDATES]
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(names)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for axis, period, title in zip(
        axes, ["historical", "holdout"], ["2022–2025", "Январь–октябрь 2026"], strict=True
    ):
        rows = metrics.loc[metrics.period == period].set_index("strategy")
        for i, name in enumerate(names):
            row = rows.loc[name]
            axis.scatter(row.es95 * 100, row.total_return * 100, s=90, color=colors[name])
            axis.annotate(
                str(i + 1),
                (row.es95 * 100, row.total_return * 100),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=11,
            )
        axis.axhline(0, color="#aaa", linewidth=0.8)
        axis.set(
            title=title, xlabel="ES95: средняя потеря в худшие 5% сессий, %", ylabel="Доходность за период, %"
        )
        axis.grid(alpha=0.2)
    handles = [
        plt.Line2D([], [], color=colors[name], marker="o", linestyle="", label=f"{i + 1}. {LABELS[name]}")
        for i, name in enumerate(names)
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("Портфель: риск и доходность после издержек")
    for suffix in ["png", "pdf"]:
        figure.savefig(ROOT / f"risk_return.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 10), layout="constrained")
    for name in names:
        curve = pd.read_csv(
            ROOT / "portfolios" / "holdout" / name / "equity.csv", index_col=0, parse_dates=True
        )
        axes[0].plot(
            curve.index,
            curve.equity / 100000,
            color=colors[name],
            label=LABELS[name],
            linewidth=2 if name == "partial_prediction" else 1,
        )
    axes[0].set(title="Капитал: январь–октябрь 2026", ylabel="Начальный капитал = 1")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    axes[0].legend(ncol=3, frameon=False, fontsize=9)
    rows = (
        comparisons.loc[(comparisons.period == "holdout") & (comparisons.block == 10)]
        .set_index("comparator")
        .loc[CANDIDATES]
    )
    values = rows.difference.to_numpy() * 100
    errors = np.array([rows.difference - rows.lower, rows.upper - rows.difference]) * 100
    axes[1].errorbar(values, np.arange(len(rows)), xerr=errors, fmt="o", color="#215ca0", capsize=4)
    axes[1].axvline(0, color="#888", linewidth=0.8)
    axes[1].set(
        yticks=np.arange(len(rows)),
        yticklabels=[LABELS[name] for name in rows.index],
        xlabel="ES95 метода − ES95 альтернативы, п.п. (слева лучше для метода)",
        title="Парные блочные интервалы; Bonferroni на 6 сравнений; блок 10 сессий",
    )
    axes[1].invert_yaxis()
    for axis in axes:
        axis.grid(alpha=0.2)
    for suffix in ["png", "pdf"]:
        figure.savefig(ROOT / f"portfolio_comparison.{suffix}", dpi=180)
    plt.close(figure)

    columns = ["strategy", *PERCENT_COLUMNS]
    periods = {
        period: metrics.loc[metrics.period == period].set_index("strategy").loc[list(LABELS)].reset_index()
        for period in ["validation", "historical", "holdout"]
    }
    interval_table = comparisons.copy()
    interval_table["comparator"] = interval_table.comparator.map(LABELS)
    for name in ["difference", "lower", "upper"]:
        interval_table[name] = interval_table[name].map(lambda value: f"{value * 100:+.3f}")
    interval_table = interval_table.drop(columns="interval_confidence").rename(
        columns={
            "period": "Период",
            "comparator": "Альтернатива",
            "block": "Блок, сессий",
            "difference": "Разность ES, п.п.",
            "lower": "Нижняя граница",
            "upper": "Верхняя граница",
        }
    )
    validation_flags = eligibility.copy()
    validation_flags.strategy = validation_flags.strategy.map(LABELS)
    validation_flags.validation_return = validation_flags.validation_return.map(lambda value: f"{value:.2%}")
    validation_flags.validation_exposure = validation_flags.validation_exposure.map(
        lambda value: f"{value:.2%}"
    )
    validation_flags.eligible = validation_flags.eligible.map({True: "да", False: "нет"})
    validation_flags.columns = ["Альтернатива", "Доходность 2021", "Доля в рынке 2021", "Прошла правило"]
    report = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Риск и доходность: Partial multistep prediction и альтернативы</title>
<style>body{{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203040;background:#fafcfe}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dce3ea;text-align:right}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}.panel{{background:white;border:1px solid #dce3ea;border-radius:10px;padding:24px}}img{{width:100%}}pre{{background:#edf2f7;padding:20px;overflow:auto}}a{{color:#145fa1}}</style>
<h1>Риск относительно других прогнозных стратегий</h1>
<p>Partial multistep prediction (Gromov–Baranov): дневная торговая адаптация на акциях, облигациях и золоте.</p>
<div class="panel"><strong>В 2026 у метода ниже наблюдаемый риск, чем у шести альтернатив.
Устойчивое преимущество между периодами не установлено.</strong>
<p>Январь–октябрь 2026: доходность метода +2,25%, ES95 0,866%, просадка 3,15%.
SMA и momentum заработали больше (+3,22% и +3,90%), но имели ES95 1,748% и 1,504%,
просадку 9,00% и 6,77%. Все четыре ML-модели за этот период убыточны и имеют более высокие ES, просадку и волатильность.</p>
<p>На 2022–2025 годах картина другая: Random Forest дал +8,17% против +0,55% у метода,
ES95 0,754% против 0,944% и просадку 6,26% против 12,02%.</p></div>
<p><a href="../report.html">Исходный эксперимент и учёт сделок</a> · <a href="portfolio_metrics.csv">Метрики портфелей</a> ·
<a href="instrument_metrics.csv">Все инструменты и методы</a> · <a href="comparisons.csv">Интервалы</a> ·
<a href="../../../../research/trading/noncrypto/predictive_protocol.json">Протокол</a> ·
<a href="frozen.json">Хеши входов и кода</a> · <a href="artifact_audit.json">Аудит</a></p>
<h2>Честный смысл «доходного метода»</h2>
<p>Шесть кандидатов заданы до их нового сравнения. Прибыльность проверяется только на 2021 году:
чистая доходность портфеля выше нуля и средняя фактическая доля капитала в рынке не менее 10%.
Все шесть прошли правило; флаги и их хеш сохранены до тестовых бэк-тестов. Убыточные строки теста
не исключены. Положительная доходность одного года не доказывает положительное математическое ожидание.</p>
{validation_flags.to_html(index=False, border=0)}
<p>Это исследовательское приложение, добавленное после знакомства с исходными результатами вне криптовалют.
Периоды 2022–2025 и январь–октябрь 2026 не объявляются новым независимым подтверждающим тестом.
Настройки и рынки по их новым результатам не менялись, гиперпараметры альтернатив не подбирались.</p>
<h2>Сравнение портфелей: январь–октябрь 2026</h2>{table(periods["holdout"][columns])}
<p>ES95 — средняя потеря в худших ceil(5% × n) торговых сессиях, минимум ноль; меньше — лучше.
Волатильность годовая, просадка по закрытиям. Доходность за весь указанный период после 15 bps издержек
с каждой стороны и с дивидендами; без налогов и процентов на cash. Для 2026 имеется 209 сессий.</p>
<p>Доля капитала в рынке у метода 43,43%: выше четырёх ML-моделей (32,52–38,21%) и momentum (41,06%),
ниже SMA (54,01%). Поэтому все различия нельзя объяснить только меньшей экспозицией метода.
Это также не доказывает способность распознавать риск: состав позиций и их время различаются.</p>
<img src="risk_return.png" alt="Риск и доходность семи стратегий на двух периодах">
<h2>Что именно суммируется</h2>
<p>Начальный капитал распределён по 1/15 между AAPL, MSFT, XOM, SPY и QQQ; по 1/3 — между TLT и GLD.
Каждая стратегия торгует в семи отдельных субсчетах. Между инструментами капитал не перераспределяется;
их веса дрейфуют. Общий капитал равен сумме стоимостей субсчетов, доходности и ES вычисляются уже из этой суммы.
Доля в рынке взвешена текущей стоимостью субсчетов. Каждый оцениваемый период начинается с нового капитала и
денежной позиции, позиции закрываются на последнем Open; это отдельные эксперименты, не одна непрерывная инвестиция.</p>
<p>Это риск портфеля с учётом совместного движения активов. В исходном отчёте оценивалось среднее
инструментальных ES с равными весами классов. Эти два показателя различаются; их численные значения нельзя подменять друг другом.</p>
<h2>Неопределённость сравнения</h2>
<p>Парный круговой блочный bootstrap использует 10 000 выборок и одинаковые сессии у всех стратегий.
Основной блок — 10 сессий; 5 и 20 — заранее заданные проверки чувствительности. Для шести сравнений
применена поправка Bonferroni: приблизительные интервалы 99,167% каждый, семейный уровень 95% для одного периода и блока.
Это не общая поправка на все исследования репозитория, оба периода или все длины блока.</p>
<p>В 2026 основной интервал исключает ноль только против SMA: разность ES −0,882 п.п.,
интервал [−1,737; −0,054]. При блоке 20 он включает ноль. Остальные пять основных интервалов тоже включают ноль.
В 2022–2025 преимущество Random Forest по ES видно при блоках 5/10, но не отделяется от нуля при 20.
Уверенного устойчивого превосходства по риску эти данные не устанавливают.</p>
<img src="portfolio_comparison.png" alt="Капитал и интервалы разности хвостового риска">
<div class="scroll">{interval_table.to_html(index=False, border=0)}</div>
<h2>Сравнение портфелей: 2022–2025</h2>{table(periods["historical"][columns])}
<h2>Все портфели на validation 2021</h2>{table(periods["validation"][columns])}
<h2>Одинаковые условия прогнозирования</h2>
<p>Random Forest, HistGradientBoosting, Ridge и SVR предсказывают доходность Close через пять сессий.
Признаки: log-momentum 1/5/20/60, волатильность log-return 20/60 и объём относительно среднего за 20 сессий.
Поквартальное обучение использует предыдущие 2520 сессий; последние пять origins исключены, чтобы все метки
предшествовали началу квартала. StandardScaler для Ridge/SVR обучается только на train. Параметры полностью
перечислены в протоколе; настроек по тесту нет. Это фиксированные представители семейств, не лучшие возможные версии моделей.</p>
<p>SMA50/200 и momentum90 — две стратегии прогноза направления по тренду, а не обучаемые регрессоры.
Buy-and-hold, сглаженный причинный контроль и контроль с геометрией статьи добавлены для контекста.
Основная модель использует ранее выбранные настройки мотивов/Wishart и отказов; дневной train и горизонт — явная адаптация.
Торговое правило отсутствует в исходной статье и не выдается за её результат.</p>
<p>Решения принимаются после последнего закрытия каждой недели, исполняются на следующем реальном Open.
Для регрессоров и основного метода прогноз роста выше 0,3% даёт целевую позицию 95%, иначе cash;
позиция держится до смены сигнала. У трендовых стратегий собственное фиксированное условие направления.
Дивиденды, сплиты, праздничные сессии и допущения поставщика совпадают с исходным экспериментом.</p>
<h2>Все инструменты: январь–октябрь 2026</h2>{table(instruments.loc[instruments.period == "holdout", ["symbol", *columns]])}
<p>Полная таблица, включая 2021 и 2022–2025, сохранена в <a href="instrument_metrics.csv">instrument_metrics.csv</a>.
Панель состоит из известных сохранившихся инструментов; результат не распространяется автоматически на все акции.
Допущения о выплатах дивидендов, издержках и бесплатном cash сохраняются. Дополнительных тестов издержек
для всех новых ML-моделей в этом приложении нет.</p>
<h2>Проверка и воспроизведение</h2><p>Повторно проверены {audit["backtests_verified"]} бэк-тестов,
{audit["portfolios_verified"]} портфелей, структура и причинные границы {audit["forecast_rows_verified"]} прогнозов.
Сохранённые сделки сверены с Open, комиссиями и дивидендами; портфельные кривые пересобраны из субсчетов,
риск независимо пересчитан. {audit["shared_instrument_curves_exactly_match_main"]} общих кривых совпали с исходным экспериментом точно.
Read-only проверка GPT Astra подтверждает порядок validation/test и способ сборки портфелей.</p>
<p><a href="cleanroom_reproduction.json">Отдельный повторный запуск</a> заново обучил все четыре ML-модели
без их прогнозных кэшей. Максимальное отличие прогнозной доходности — ≈1,04 × 10⁻¹⁷; targets, сделки,
кривые капитала, метрики и bootstrap-интервалы воспроизвелись точно при численном чтении CSV.
Проверены 1128 CSV-файлов. Основное исследование отдельно воспроизведено из сохранённых прогнозов и снимков данных.
<a href="../verification.json">Общая проверка</a>: 92 теста пройдены, lint и форматирование пройдены.</p>
<pre>bash research/trading/noncrypto/reproduce.sh</pre>
<p>Сценарий восстанавливает сохранённые снимки данных и запускает оба исследования, аудит и отчёты.
Чтобы заново обучить четыре альтернативных ML-модели, удалите их кэши predictive/*/forecasts.csv
в отдельной копии репозитория перед запуском. Хеши исходного кода и протокола должны совпасть.</p>
</html>"""
    (ROOT / "report.html").write_text(report)
    summary = dict(
        conclusion="Lower observed 2026 risk than six predictive alternatives; no stable cross-period advantage established",
        eligibility=eligibility.to_dict("records"),
        holdout=periods["holdout"].to_dict("records"),
        historical=periods["historical"].to_dict("records"),
        primary_intervals=comparisons.loc[comparisons.block == 10].to_dict("records"),
        exploratory=True,
        portfolio_es_not_mean_instrument_es=True,
        audit=audit,
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(ROOT / "report.html")


if __name__ == "__main__":
    build_report()
