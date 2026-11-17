## Lorenz

| Horizon | Paper RMSE | Paper-parameter RMSE | Custom-parameter RMSE |
| --- | --- | --- | --- |
| 1 | 0.01 | 0.026021 | 0.046810 |
| 10 | 0.03 | 0.075502 | 0.081152 |
| 50 | 0.04 | 0.261395 | 0.254824 |
| 100 | 0.04 | 0.245543 | 0.227748 |

## German electricity load

| Horizon | Paper RMSE | Paper-parameter RMSE | Custom-parameter RMSE |
| --- | --- | --- | --- |
| 1 | 0.01 | 0.078322 | 0.029688 |
| 10 | 0.01 | 0.115260 | 0.064532 |
| 50 | 0.05 | 0.234916 | 0.055781 |
| 100 | 0.03 | 0.219611 | 0.050491 |

| Directory | Contents |
| --- | --- |
| src/ | Forecasting methods; Forecaster(**parameters).fit(train).predict(context, horizon=100) |
| data/ | Hourly German electricity load and source metadata |
| references/ | Original paper and published metrics |
| results/ | Configurations, predictions and metrics |
| trading-research/data/ | Frozen market data, forecasts and original experiment provenance |
| trading-research/tools/ | Forecasting, comparator models, execution and report tools |
| trading-research/results/ | Separate generated HTML reports and numerical results for each market block |

| Market block | Run |
| --- | --- |
| Bitcoin | `uv run --extra trading python trading-research/btc.py` |
| Other crypto | `uv run --extra trading python trading-research/crypto.py` |
| Stocks, bonds and gold | `uv run --extra trading python trading-research/noncrypto.py` |

Parameters are defined in each script’s `CONFIG`. Trading calls the unchanged `src.Forecaster.fit/predict`; identical inputs reuse its computed cache. Each script prints its HTML report path.
