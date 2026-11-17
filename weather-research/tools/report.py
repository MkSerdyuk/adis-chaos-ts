import json
from html import escape

import pandas as pd
from bokeh.embed import components
from bokeh.layouts import column
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.resources import INLINE

from .baselines import NAMES


def write_report(output, predictions, measured, intervals, config):
    sections = []
    primary = intervals[intervals.primary]
    sections.append(
        "<h2>Primary comparison: 2026, 24 hours</h2>"
        + primary[
            ["comparator", "difference", "lower", "upper", "accepted_per_station", "finite_replicates"]
        ].to_html(index=False, float_format=lambda v: f"{v:.4f}")
    )
    coverage = measured[
        (measured["mask"] == "available") & measured.method.isin(["forecaster", "paper_geometry"])
    ]
    sections.append(
        "<h2>Coverage: 2026, 24 hours</h2>"
        + coverage[(coverage.period == "holdout") & (coverage.horizon == 24)][
            [
                "period",
                "station",
                "method",
                "horizon",
                "scheduled_targets",
                "eligible_targets",
                "count",
                "coverage",
            ]
        ].to_html(index=False, float_format=lambda v: f"{v:.4f}")
    )
    for station, name in config["stations"].items():
        rows = predictions[
            (predictions.station == station)
            & (predictions.horizon == 24)
            & (predictions.origin >= pd.Timestamp(config["periods"]["holdout"][0], tz="UTC"))
            & (predictions.target_time < pd.Timestamp(config["periods"]["holdout"][1], tz="UTC"))
            & predictions.eligible
        ]
        plot = figure(
            title=f"{name}: 24-hour forecasts, 2026",
            x_axis_type="datetime",
            height=400,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        truth = rows[rows.method == "forecaster"].sort_values("target_time")
        plot.line(
            truth.target_time.dt.tz_localize(None),
            truth.truth,
            line_color="black",
            legend_label="Observed",
            line_width=3,
        )
        for i, method in enumerate(["forecaster", *NAMES]):
            frame = rows[rows.method == method].sort_values("target_time")
            plot.line(
                frame.target_time.dt.tz_localize(None),
                frame.prediction,
                color=Category10[10][i],
                legend_label=method,
                line_width=2 if method == "forecaster" else 1,
            )
        plot.yaxis.axis_label = "Temperature (°C)"
        plot.legend.click_policy = "hide"
        plot.add_layout(plot.legend[0], "right")
        error = figure(
            title=f"{name}: MAE on accepted Forecaster forecasts, 2026",
            height=320,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,reset,save",
        )
        table = measured[
            (measured.station == station)
            & (measured.period == "holdout")
            & (measured["mask"] == "forecaster")
        ]
        for i, method in enumerate(["forecaster", *NAMES]):
            frame = table[table.method == method].sort_values("horizon")
            error.line(frame.horizon, frame.mae, color=Category10[10][i], legend_label=method, line_width=2)
            error.scatter(frame.horizon, frame.mae, color=Category10[10][i], size=6)
        error.xaxis.axis_label = "Forecast horizon (hours)"
        error.yaxis.axis_label = "MAE (°C)"
        error.legend.click_policy = "hide"
        error.add_layout(error.legend[0], "right")
        script, div = components(column(plot, error, sizing_mode="stretch_width"))
        sections.append(f"<h2>{escape(name)}</h2>{div}{script}")
    sections.append(
        "<details><summary>Coverage at every horizon</summary>"
        + coverage[
            [
                "period",
                "station",
                "method",
                "horizon",
                "scheduled_targets",
                "eligible_targets",
                "count",
                "coverage",
            ]
        ].to_html(index=False, float_format=lambda v: f"{v:.4f}")
        + "</details>"
    )
    for mask, label in [
        ("forecaster", "Paired comparisons on accepted Forecaster forecasts"),
        ("paper_geometry", "Paired comparisons on accepted paper-geometry forecasts"),
        ("available", "Available forecasts: inspect coverage before comparing"),
    ]:
        frame = measured[measured["mask"] == mask]
        sections.append(
            f"<h2>{label}</h2>"
            + frame.drop(columns="mask").to_html(index=False, float_format=lambda v: f"{v:.4f}")
        )
    sections.append(
        "<details><summary>Bootstrap sensitivity</summary>"
        + intervals.to_html(index=False, float_format=lambda v: f"{v:.5f}")
        + "</details>"
    )
    state = json.loads((output / "run.json").read_text())
    sections.append(
        "<details><summary>Configuration and provenance</summary><pre>"
        + escape(json.dumps(state, indent=2))
        + "</pre></details>"
    )
    report = output / "report.html"
    report.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Weather research</title>
    {INLINE.render()}<style>body{{font:15px system-ui;margin:32px;max-width:1600px}}table{{border-collapse:collapse;font-size:13px}}
    th,td{{padding:6px;border-bottom:1px solid #ddd;text-align:right}}h2{{margin-top:40px}}pre{{white-space:pre-wrap}}details{{margin:24px 0;overflow:auto}}</style>
    <h1>Weather research: 2 m air temperature</h1>
    <p>Unmodified src.Forecaster. Fixed parameters transferred from the legacy BTC selection; separate paper-geometry control.
    Validation: 2021. Historical test: 2022–2025. Historical holdout: January–October 2026. No weather tuning.
    Quarterly training on the preceding 10000 hours; daily forecasts issued after the 23:00 UTC observation.</p>
    <p>MAE, RMSE, bias and tail95 are in °C. Tail95 is the mean largest 5% of absolute errors.
    Primary differences are equal-station MAE(Forecaster) minus MAE(comparator), on the same accepted targets.
    Negative differences favor Forecaster. Intervals use synchronized 14-day blocks and Bonferroni correction for seven comparisons.
    Coverage is accepted forecasts divided by eligible observed targets. Eligibility requires the current and preceding 168 hourly observations. Forward-fill is confined to model inputs; verification targets remain original observations. Available-case scores with different coverage are not a ranking of overall forecasting quality.</p>
    <p>Source: <a href="https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/">Deutscher Wetterdienst, CDC</a>.
    Frozen historical data may contain retrospective corrections; recent 2026 data are provisional. This is a station-temperature benchmark,
    not an evaluation of operational publication delays or numerical weather prediction systems. Horizons also correspond to different times of day.</p>
    {"".join(sections)}</html>""")
    return report
