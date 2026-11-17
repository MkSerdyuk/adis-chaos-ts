import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from .backtest import expected_shortfall, run_backtest
from .data import load_market
from .forecasts import fingerprint, source_hashes, walk_forward

ROOT = Path("results/trading")
PROTOCOL = Path("research/trading/protocol.json")
PAPER = dict(
    length=4,
    max_gap=10,
    r=11,
    mu=0.2,
    epsilon=0.01,
    cluster_eps=0.01,
    min_samples=3,
    perturbation_variance=1e-5,
    paths=16,
    training_hours=10000,
)


def experiment_identity():
    sources = source_hashes()
    for name in ["backtest.py", "baselines.py", "experiment.py"]:
        sources[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return dict(
        source_hashes=sources,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(Path("data/trading/BTCUSDT_1h.csv.gz").read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path("data/trading/manifest.json").read_bytes()).hexdigest(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "numba", "scikit-learn", "backtesting", "optuna"]
        },
    )


def suggest_parameters(trial):
    geometry = trial.suggest_categorical("geometry", ["L3K3", "L4K3", "L4K5", "L4K10", "L5K3"])
    length, gap = geometry[1:].split("K")
    return dict(
        length=int(length),
        max_gap=int(gap),
        r=trial.suggest_categorical("r", [5, 11, 21]),
        mu=trial.suggest_categorical("mu", [0.02, 0.2, 2.0]),
        epsilon=trial.suggest_float("epsilon", 0.005, 0.2, log=True),
        cluster_eps=trial.suggest_float("cluster_eps", 0.001, 0.05, log=True),
        min_samples=trial.suggest_categorical("min_samples", [1, 3, 5]),
        perturbation_variance=trial.suggest_float("perturbation_variance", 1e-7, 1e-4, log=True),
        paths=trial.suggest_categorical("paths", [8, 16]),
        training_hours=10000,
    )


def tune(frame, trials=32):
    ROOT.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=20261117, n_startup_trials=8, constraints_func=lambda trial: trial.user_attrs["constraints"]
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="btc_risk_validation_v3",
        storage=f"sqlite:///{ROOT / 'study.db'}",
        load_if_exists=True,
    )
    if not study.trials:
        paper = {
            key: value for key, value in PAPER.items() if key not in {"length", "max_gap", "training_hours"}
        }
        study.enqueue_trial(dict(geometry="L4K10", **paper))
        study.enqueue_trial(dict(geometry="L4K10", **(paper | {"epsilon": 0.05})))
    identity = experiment_identity()
    if "identity" in study.user_attrs:
        assert study.user_attrs["identity"] == identity, "Existing study uses different inputs or code"
    else:
        assert not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        study.set_user_attr("identity", identity)

    def objective(trial):
        parameters = suggest_parameters(trial)
        ledger = walk_forward(frame, "2021-01-01", "2022-01-01", parameters, ROOT)
        result = run_backtest(frame, ledger.target, "2021-01-01", "2022-01-01")
        returns = result["daily_returns"]
        quarterly_es, quarterly_exposure = [], []
        for start in pd.date_range("2021-01-01", "2021-10-01", freq="QS", tz="UTC"):
            end = start + pd.DateOffset(months=3)
            quarterly_es.append(
                expected_shortfall(returns.loc[(returns.index >= start) & (returns.index < end)])
            )
            exposure = result["exposure"].loc[
                (result["exposure"].index >= start) & (result["exposure"].index < end)
            ]
            quarterly_exposure.append(float(exposure.mean()))
        metrics = result["metrics"]
        constraints = [
            (0.10 - metrics["mean_exposure"]) / 0.10,
            (12 - metrics["entries"]) / 12,
            (0.02 - min(quarterly_exposure)) / 0.02,
        ]
        score = float(np.mean(quarterly_es) + 0.5 * np.std(quarterly_es))
        trial.set_user_attr("constraints", constraints)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("quarterly_es", quarterly_es)
        trial.set_user_attr("quarterly_exposure", quarterly_exposure)
        trial.set_user_attr("coverage", float(ledger.accepted.mean()))
        trial.set_user_attr("ledger_key", fingerprint(parameters))
        folder = ROOT / "validation" / f"trial_{trial.number:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(folder / "predictions.csv")
        result["trades"].to_csv(folder / "trades.csv", index=False)
        pd.DataFrame(dict(equity=result["equity"], exposure=result["exposure"])).to_csv(
            folder / "equity.csv.gz"
        )
        (folder / "result.json").write_text(
            json.dumps(
                dict(
                    parameters=parameters,
                    metrics=metrics,
                    constraints=constraints,
                    quarterly_es=quarterly_es,
                    score=score,
                ),
                indent=2,
            )
        )
        return score

    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    study.optimize(objective, n_trials=max(0, trials - completed), gc_after_trial=True)
    done = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    feasible = [trial for trial in done if max(trial.user_attrs["constraints"]) <= 0]
    if feasible:
        winner = min(
            feasible,
            key=lambda trial: (trial.value, -trial.user_attrs["metrics"]["total_return"], trial.number),
        )
    else:
        winner = min(
            done,
            key=lambda trial: (
                sum(max(0, value) for value in trial.user_attrs["constraints"]),
                trial.value,
                trial.number,
            ),
        )
    selected = dict(
        trial=winner.number,
        parameters=winner.user_attrs["parameters"],
        eligible=bool(feasible),
        validation_metrics=winner.user_attrs["metrics"],
        constraints=winner.user_attrs["constraints"],
        objective=winner.value,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        selected_before_test=True,
        completed_trials=len(done),
        identity=identity,
    )
    (ROOT / "selected.json").write_text(json.dumps(selected, indent=2))
    study.trials_dataframe().to_csv(ROOT / "optimization_trials.csv", index=False)
    print(json.dumps(selected, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["tune"])
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    set_num_threads(args.threads)
    with threadpool_limits(limits=1):
        tune(load_market(), args.trials)


if __name__ == "__main__":
    main()
