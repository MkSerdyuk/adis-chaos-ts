import json
from html import escape

import pandas as pd
from bokeh.embed import components
from bokeh.layouts import column
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.resources import INLINE


def record(results, result, **labels):
    if labels["scenario"] in {"sensitivity", "seed"}:
        results.append((labels, {"metrics": result["metrics"]}))
        return result
    result.pop("engine", None)
    result.pop("stats", None)
    if "curve" not in result:
        result["curve"] = pd.DataFrame(dict(equity=result.pop("equity"), exposure=result.pop("exposure")))
        result["returns"] = result.pop("daily_returns")
    results.append((labels, result))
    return result


def write_report(output, title, results, tables, notes):
    metrics = pd.DataFrame(
        [dict(run=i, **labels, **result["metrics"]) for i, (labels, result) in enumerate(results)]
    )
    metrics.to_csv(output / "runs.csv", index=False)
    for name in ["curve", "returns", "trades", "orders", "dividends"]:
        frames = []
        for i, (_, result) in enumerate(results):
            if name in result and len(result[name]):
                frame = result[name]
                if isinstance(frame, pd.Series):
                    frame = frame.rename("return").to_frame()
                if name in {"curve", "returns"}:
                    frame = frame.rename_axis("time").reset_index()
                frames.append(frame.assign(run=i))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.csv.gz", index=False)
    groups = {}
    for labels, result in results:
        if labels["scenario"] in {"main", "predictive", "portfolio"}:
            key = (labels["scenario"], labels["symbol"], labels["period"])
            groups.setdefault(key, []).append((labels["strategy"], result["curve"]))
    sections = []
    for (scenario, symbol, period), values in groups.items():
        plots = [
            figure(
                title=f"{symbol} · {period} · {scenario} · {label}",
                x_axis_type="datetime",
                height=380,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            for label in ["Equity", "Drawdown"]
        ]
        for i, (name, curve) in enumerate(values):
            equity = curve.equity.resample("1D").last().dropna()
            drawdown = (
                (1 - curve.equity / curve.equity.cummax().clip(lower=100000)).resample("1D").max().dropna()
            )
            for plot, y in zip(plots, [equity, drawdown], strict=True):
                plot.line(
                    equity.index.tz_localize(None),
                    y.to_numpy(),
                    legend_label=name,
                    color=Category20[20][i % 20],
                    line_width=2,
                )
        for plot in plots:
            plot.legend.click_policy = "hide"
            plot.add_layout(plot.legend[0], "right")
        script, div = components(column(*plots, sizing_mode="stretch_width"))
        subset = metrics[
            (metrics.scenario == scenario) & (metrics.symbol == symbol) & (metrics.period == period)
        ]
        columns = ["strategy", "total_return", "es95", "max_drawdown", "mean_exposure", "final_equity"]
        table = (
            subset[columns]
            .rename(
                columns=dict(
                    strategy="Method",
                    total_return="Net return",
                    es95="ES95",
                    max_drawdown="Max drawdown",
                    mean_exposure="Mean exposure",
                    final_equity="Final equity",
                )
            )
            .to_html(index=False, float_format=lambda value: f"{value:.6f}")
        )
        sections.append(f"<h2>{escape(symbol)} / {escape(period)} / {scenario}</h2>{table}{div}{script}")
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
        sections.append(
            f"<h2>{escape(name.replace('_', ' ').title())}</h2>"
            + frame.to_html(index=False, float_format=lambda value: f"{value:.8f}")
        )
    summary = metrics.groupby("scenario").size().to_dict()
    (output / "completed.json").write_text(
        json.dumps(
            dict(
                backtests=sum(value for name, value in summary.items() if name != "portfolio"),
                portfolios=summary.get("portfolio", 0),
                scenarios=summary,
            ),
            indent=2,
        )
        + "\n"
    )
    sections.append("<details><summary>All backtests</summary>" + metrics.to_html(index=False) + "</details>")
    run = json.loads((output / "run.json").read_text())
    sections.append(
        "<details><summary>Run configuration and provenance</summary><pre>"
        + escape(json.dumps(run, indent=2))
        + "</pre></details>"
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(title)}</title>
    {INLINE.render()}<style>body{{font:15px system-ui;margin:32px;max-width:1600px}}table{{border-collapse:collapse;font-size:13px}}
    th,td{{padding:6px;border-bottom:1px solid #ddd;text-align:right}}h2{{margin-top:40px}}pre{{white-space:pre-wrap}}
    details{{margin:24px 0;overflow:auto}}</style><h1>{escape(title)}</h1><p>{escape(notes)}</p>
    <p>Forecast source: {run["forecast_source"]}. Backtests: {sum(value for name, value in summary.items() if name != "portfolio")}. Portfolios: {summary.get("portfolio", 0)}.
    Run: {output.name}. ES95 differences above zero mean higher risk for the partial prediction method.</p>
    {"".join(sections)}</html>"""
    report = output / "report.html"
    report.write_text(html)
    return report
