from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator

from .motifs import MotifModel
from .prediction import ForecastConfig, ensemble_forecast


@dataclass
class Forecaster(BaseEstimator):
    length: int = 4
    max_gap: int = 10
    r: int = 11
    mu: float = 0.2
    clustered: bool = True
    patterns: object = None
    epsilon: float = 0.01
    cluster_eps: float = 0.01
    min_samples: int = 3
    trajectories: int = 16
    noise_variance: float = 1e-5
    endpoint_eps: float = 0.01
    endpoint_min_samples: int = 1
    decay: float = 0.99
    seed: int = 42
    noise_schedule: str = "fixed"
    noise_law: str = "normal"
    growth_rule: str = "strict"

    def fit(self, series, y=None, workers=4):
        self.model_ = MotifModel(
            self.length, self.max_gap, self.r, self.mu, self.clustered, self.patterns
        ).fit(series, workers=workers)
        self.context_size_ = self.model_.context_size
        return self

    def predict(self, context, horizon=100, origin_ids=None):
        context = np.asarray(context, dtype=float)
        single = context.ndim == 1
        contexts = np.atleast_2d(context)
        origins = np.arange(len(contexts)) if origin_ids is None else np.asarray(origin_ids)
        config = ForecastConfig(
            aggregation="avg",
            selection="rgdbscan",
            epsilon=self.epsilon,
            cluster_eps=self.cluster_eps,
            min_samples=self.min_samples,
            perturbation_variance=self.noise_variance,
            decay=self.decay,
            seed=self.seed,
        )
        prediction = ensemble_forecast(
            self.model_,
            contexts,
            horizon,
            config,
            self.trajectories,
            self.endpoint_eps,
            self.endpoint_min_samples,
            self.noise_schedule,
            self.noise_law,
            origins,
            self.growth_rule,
        )
        return prediction[0] if single else prediction
