# Appliance Energy Forecasting

Day ahead forecasting of household appliance energy consumption, comparing
benchmark methods, a seasonal ARIMA model, gradient boosted feature models and
a pretrained time series foundation model under a single evaluation protocol.

The headline result is negative and deliberately so. The best model that uses
only information available when the forecast is issued improves on the strongest
benchmark by 18.9% in MASE, but a Diebold and Mariano test cannot distinguish that
improvement from zero (p = 0.083). The only model that beats the benchmark at
the 5% level is one built on features that would not exist in operation.

---

## Dataset

Appliances Energy Prediction, UCI Machine Learning Repository dataset 374.

| Property | Value |
|---|---|
| Source | Low energy house in Stambruges, Belgium |
| Raw resolution | 10 minutes |
| Period | 11 January to 27 May 2016 (137 days) |
| Raw observations | 19,735 |
| Hourly observations after aggregation | 3,290 |
| Target | `Appliances`, energy use in Wh |
| Covariates | 8 indoor temperature sensors, 8 indoor humidity sensors, 8 outdoor weather variables |

Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017). Data driven prediction
models of energy use of appliances in a low energy house. *Energy and
Buildings*, 140, 81-97.

---

## Results

Fourteen successive 24 hour forecasts, midnight aligned origins, expanding
training window. MASE is scaled by the in sample seasonal naive error on the
training data at the first origin, so values below 1 beat that reference.

| Model | Family | MASE | MAE (Wh) | Skill vs benchmark | DM p value |
|---|---|---|---|---|---|
| `xgb_leaky_demo` | Control, leaky | 0.621 | 32.9 | +26.8% | **0.025** |
| `xgb_full_honest_log` | Machine learning | 0.688 | 36.4 | +18.9% | 0.083 |
| `seasonal_mean_28d` | Benchmark | 0.710 | 37.6 | +16.3% | 0.178 |
| `sarima_log` | Statistical | 0.729 | 38.6 | +14.1% | 0.262 |
| `chronos_bolt` | Foundation (zero shot) | 0.716 | 37.9 | +15.6% | 0.071 |
| `xgb_calendar_lags` | Machine learning | 0.742 | 39.3 | +12.5% | 0.206 |
| `xgb_full_honest` | Machine learning | 0.756 | 40.1 | +10.9% | 0.302 |
| `xgb_full_conditional` | Machine learning | 0.766 | 40.6 | +9.7% | 0.357 |
| `rf_full_honest` | Machine learning | 0.770 | 40.8 | +9.2% | 0.348 |
| `sarima` | Statistical | 0.791 | 41.9 | +6.7% | 0.594 |
| **`seasonal_naive_weekly`** | **Benchmark** | **0.848** | **44.9** | reference | n/a |
| `naive` | Benchmark | 0.962 | 51.0 | -13.4% | 0.390 |
| `drift` | Benchmark | 0.962 | 51.0 | -13.5% | 0.389 |
| `seasonal_naive_daily` | Benchmark | 0.966 | 51.2 | -13.9% | 0.387 |
| `mean` | Benchmark | 0.986 | 52.3 | -16.3% | n/a |

`xgb_leaky_demo` is retained only as a control. It reproduces the feature
construction of the supplied demonstration pipeline, which uses lags of 1 to 12
hours and rolling windows shifted by a single step. None of those values exist
when a forecast is issued at midnight for the following day.

---

## Findings

**Weekly seasonality beats daily.** Of the four benchmarks named in the brief,
the weekly seasonal naive forecast is strongest at MASE 0.848 against 0.966 for
the daily version. The autocorrelation at lag 168 (0.304) is effectively equal
to that at lag 24 (0.305), so matching the day of the week matters as much as
matching the hour.

**Averaging beats matching.** A 28 day seasonal mean reaches MASE 0.710 without
fitting anything. The weakness of the daily seasonal naive forecast is not the
seasonal assumption but the sampling variance of relying on one noisy day.

**The series is mostly noise.** STL attributes 2.7% of variance to trend, 34.6%
to daily seasonality and 62.7% to the remainder. This ceiling explains why every
model clusters between MASE 0.69 and 0.85, and why none separates cleanly.

**Sensors and weather add nothing.** Native tree importance assigns 26% of total
gain to indoor sensors and lagged weather. Group permutation importance measured
on held out data puts them at +0.12 Wh and +0.77 Wh respectively. Adding those
28 columns moves MASE the wrong way, from 0.742 to 0.756.

**Perfect weather knowledge is worthless.** A model given the recorded outdoor
weather for the forecast window scores 0.766 against 0.756 for the model given
only lagged weather (p = 0.55). The limiting factor is occupant behaviour, which
no weather feed predicts.

**A model fitted to nothing is competitive.** Chronos-Bolt reaches MASE 0.716 zero shot, beating
SARIMA and every gradient boosted model bar one, running all fourteen windows in 0.77 seconds
against 153 seconds for SARIMA. It also produces the best calibrated intervals in the study,
80.65% empirical coverage against an 80% nominal level.

**The leading models are statistically indistinguishable.** No pairwise
Diebold and Mariano test among the top four honest models reaches significance
(p = 0.25 to 0.84). The choice between them should rest on interpretability,
uncertainty quantification and running cost.

---

## Repository layout

```
appliance-energy-forecasting/
├── data/
│   ├── raw/                  energydata_complete.csv, as downloaded
│   └── processed/            appliance_hourly.csv, built by src/data.py
├── src/
│   ├── data.py               loading, auditing, hourly aggregation, splitting
│   ├── eda.py                stationarity tests, correlograms, decomposition
│   ├── features.py           leakage safe feature construction
│   ├── evaluation.py         metrics and the rolling origin backtest engine
│   ├── plots.py              every figure in the report
│   └── models/
│       ├── sarimax.py        order selection, diagnostics, forecasting
│       ├── gradient_boosting.py   XGBoost, random forest, importance
│       └── foundation.py     Chronos-2 and Chronos-Bolt, zero shot
├── scripts/
│   ├── run_eda.py            Part 1
│   ├── grid_worker.py        Part 4a, resumable AIC grid search
│   ├── run_order_selection.py     Part 4a, single invocation version
│   ├── run_model.py          Parts 3, 4b, 6, with result caching
│   ├── run_sarimax_diagnostics.py Part 4b
│   ├── run_feature_analysis.py    Part 5
│   ├── run_foundation.py     Part 7
│   ├── run_evaluation.py     Part 8
│   └── run_pipeline.py       everything, in order
├── notebooks/
│   └── 07_foundation_model_chronos.ipynb   Colab runner for Part 7
├── outputs/
│   ├── figures/              all report figures
│   ├── forecasts/            cached per model forecasts
│   └── metrics/              all result tables
├── tests/                    correctness checks, including leakage guards
├── report/                   the written report (8 pages, docx and pdf)
├── models/                   locally supplied foundation model weights (gitignored)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repository-url>
cd appliance-energy-forecasting

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The foundation model component additionally needs `torch` and
`chronos-forecasting`, listed separately in `requirements-foundation.txt` because they are
large and the rest of the pipeline runs without them.

Where Hugging Face is unreachable, download `config.json` and `model.safetensors` from
`huggingface.co/amazon/chronos-bolt-small` and place them in `models/chronos-bolt-small/`.
The loaders check that directory before attempting any download, so nothing else changes.

---

## Reproducing the results

Run everything:

```bash
python scripts/run_pipeline.py
```

Or step by step:

```bash
python scripts/run_eda.py                    # Part 1: EDA and stationarity

python scripts/grid_worker.py --stage 1      # Part 4a: 147 candidate AIC grid
python scripts/grid_worker.py --stage 2 --order 1,0,6

python scripts/run_model.py --all            # Parts 3, 4b, 6
python scripts/run_sarimax_diagnostics.py    # Part 4b: residuals, intervals
python scripts/run_feature_analysis.py       # Part 5: importance and ablation
python scripts/run_foundation.py             # Part 7: Chronos
node scripts/build_report.js                 # build the report
python scripts/run_evaluation.py             # Part 8: comparison
```

`run_model.py` caches every backtest under `outputs/forecasts/`. Re running is
cheap; use `--force` to recompute. The grid search is resumable and can be
interrupted freely, since results are appended after each candidate.

Run the tests with:

```bash
python -m pytest tests/ -v
```

---

## Design decisions

### Evaluation: rolling origin rather than a single split

The brief asks for a 14 day test period and a 24 hour horizon. These are
reconciled by re anchoring the origin at the start of each test day and issuing
fourteen successive 24 hour forecasts, rather than one 336 step forecast.

This matches the operational question, keeps every model at the same horizon,
and yields fourteen windows instead of one. The last point matters: window level
MASE ranges from roughly 0.3 to 2.0 for every model, so a single window would
have been close to meaningless.

Origins are aligned to midnight so each window is a calendar day. The record
ends at 18:00, so alignment discards 19 trailing observations, which is
preferable to windows that begin in the middle of the evening peak.

### Features: lag 24 is the floor

A forecast issued at midnight for the following day can use the observation at
23:00 the previous evening. For the target at hour `h` of the forecast day that
observation sits at lag `h + 1`, so the shortest lag available across the whole
horizon is 24. `features.py` enforces this in code and raises `LeakageError` on
violation.

Three tiers of covariate are distinguished so that the honest and optimistic
cases can be compared directly: calendar features known indefinitely ahead,
lagged observations genuinely available, and outdoor weather during the forecast
window, which stands in for a perfect weather forecast. Indoor sensor readings
during the forecast window belong to no tier and are never used.

### Order selection: AIC comparability under a fast fitting path

Fitting is accelerated with `concentrate_scale` and `simple_differencing`, an
11-fold speedup that makes the required 147 candidate grid tractable on one core.
`simple_differencing` discards the first `d + D·s` observations, so a candidate
with heavier differencing would otherwise be scored on fewer points and its AIC
would not be comparable. The grid trims the front of the series per candidate so
that all are estimated on an identical effective sample, verified at 1,474
observations throughout.

That fast path must not be used to forecast: under `simple_differencing`,
`get_forecast` returns values on the differenced scale. On this data it yields a
mean 24 hour forecast near 13 Wh against an actual near 169 Wh.
`make_sarimax_forecaster` therefore defaults to the exact path.

### Residual diagnostics: burn in is not optional

The Kalman filter starts from a diffuse prior, so the first residuals of a
state space fit reflect the filter converging rather than any failure of the
model. Discarding the first seasonal period changes the log scale residual skew
from 3.41 to 0.70 and the excess kurtosis from 14.20 to 2.78, and changes the
Ljung and Box verdict at lag 24 from p < 0.0001 to p = 0.45. Without the burn in the
model appears to leave substantial structure uncaptured; with it, white noise is
not rejected at any lag tested.

### Target scale

Both scales were run end to end. The log target improves SARIMA from MASE 0.791
to 0.729 (p = 0.005, the only significant pairwise difference found in the whole
study) and narrows the 95% interval from 234 Wh to 155 Wh at comparable
coverage. Back transformation applies the lognormal correction
`exp(μ + σ²/2)` so the result estimates the conditional mean rather than the
median.

---

## Limitations

The test period is a single fortnight in late May. Load patterns in a Belgian
house differ between winter and spring, and the training data spans both, so
performance here need not generalise to a winter test period.

MASE values below 1 across nearly all models partly reflect the scaling
denominator being computed over a training period of higher variance than the
test period. Comparisons between models remain valid; the absolute level should
not be read as skill against a contemporaneous seasonal naive forecast.

Only one house is studied, over 137 days. Any conclusion about which model
family suits smart home forecasting rests on a single building and a single
season.

The foundation model is described as zero shot, but Chronos was pretrained on a
corpus including energy and demand series. It has not seen these observations;
it has very probably seen series of this kind.

---

## Licence

Released for academic assessment. The underlying dataset is distributed by the
UCI Machine Learning Repository under its own terms.
