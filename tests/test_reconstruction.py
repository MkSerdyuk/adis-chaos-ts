import numpy as np
import pandas as pd
import pytest

from adis.reconstruction import PROFILES, check_frozen, freeze, ranked, score


def test_search_constraints_require_coverage_at_every_late_horizon():
    truth = np.ones((100, 100))
    prediction = truth.copy()
    prediction[15:, 70] = np.nan
    scores = pd.DataFrame([{"trial": 0} | score(truth, prediction)])
    assert scores.coverage.iloc[0] > 0.99
    assert not ranked(scores, "lorenz_rk4").feasible.iloc[0]


def test_search_does_not_reward_missing_an_entire_horizon():
    truth = np.ones((100, 100))
    prediction = truth.copy()
    prediction[:, 30] = np.nan
    scores = pd.DataFrame([{"trial": 0} | score(truth, prediction)])
    assert np.isinf(scores.macro_rmse.iloc[0])
    assert not ranked(scores, "lorenz_rk4").feasible.iloc[0]


def test_search_chooses_feasible_accuracy_before_low_coverage_accuracy():
    rows = pd.DataFrame(
        [
            {
                "trial": 0,
                "macro_rmse": 0.01,
                "coverage": 0.1,
                "early_coverage": 0.1,
                "minimum_late_coverage": 0.1,
            },
            {
                "trial": 1,
                "macro_rmse": 0.03,
                "coverage": 0.6,
                "early_coverage": 0.95,
                "minimum_late_coverage": 0.5,
            },
        ]
    )
    result = ranked(rows, "germany")
    assert result.iloc[0].trial == 1
    assert result.iloc[0].feasible


@pytest.mark.parametrize("changed", ["selection.json", "data.json", "protocol.json"])
def test_evaluation_rejects_changes_after_freezing(tmp_path, changed):
    (tmp_path / "protocol.json").write_text('{"rule": "validation only"}')
    for profile in PROFILES:
        directory = tmp_path / profile
        directory.mkdir()
        (directory / "selection.json").write_text('{"epsilon": 0.01}')
        (directory / "data.json").write_text('{"train": [0, 10000]}')
    freeze(tmp_path)
    directory = tmp_path / PROFILES[0]
    check_frozen(directory)
    path = (tmp_path if changed == "protocol.json" else directory) / changed
    path.write_text("{}")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(directory)
