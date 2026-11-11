# EPİAŞ seasonal-naive adequacy diagnostic (Phase 11 closeout)

Based exclusively on user-supplied results for EPİAŞ PTF, training regime 10%, target-only iTransformer, 96→24 forecasting and 830 matched optimizer steps.

## Numeric findings

- 3 model initializations × (3 real controls + 2 VAE pools) = **15 learned fits**. **15/15** have test MSE and MAE above a fixed daily seasonal-naive forecast using the last 24 observed target values to predict the next 24 steps.
- Seasonal naive: normalized **MSE 2.73123264**, **MAE 1.01220262**.
- Best learned normalized MSE: step-only baseline rep 1 **4.859287** (approximately **+77.9155%** worse).
- Best learned normalized MAE: step-only baseline rep 1 **1.686465** (approximately **+66.61%** worse).
- Duplicated-real paired comparison: VAE worse in **6/6**; step-only baseline paired comparison: VAE worse in **5/6**; VAE beats both matched controls in **0/6**.
- Hierarchical pool-then-model-seed median VAE change vs duplicate-real MSE: **+9.6257%**; vs equal-step real-only: **+4.6469%**. These are descriptive medians, not confidence intervals.
- Training target std in original units 464.1324; standardized test std **2.71026** and test mean **−0.76761**. These are consistent with distribution change but do not identify its cause.
- All data-quality preflight checks passed: 39,552 hourly records, zero missing or irregular naive-local-hour intervals; original source equals prepared array under existing data pipeline, SHA stored in Phase 11 log.

## Proposed Results text (English)

> In the EPİAŞ day-ahead electricity price dataset, the 10% training-regime evaluation did not show an augmentation advantage. Across three model initializations and two independently generated VAE pools, VAE augmentation underperformed the duplicated-real control in all six matched comparisons; the hierarchical median test-MSE difference was +9.63%. VAE exceeded the real-only equal-step baseline in test MSE in five of six matched comparisons, with a median difference of +4.65%. As an additional forecasting-adequacy diagnostic, a daily seasonal-naive predictor achieved a normalized test MSE of 2.731, lower than every neural-model result across the 15 fits (best neural MSE: 4.859). Accordingly, the EPİAŞ experiment should be interpreted as a within-protocol augmentation contrast rather than as evidence of competitive absolute forecasting performance. The selected-training-slice and test-target distributions also differed substantially (test standard deviation 2.71 in training-normalized units), but this observation does not establish that a distribution shift caused the outcome.

## Reviewer response (English)

> We thank the reviewer for raising the importance of genuine forecasting baselines. Beyond the strict matched-optimizer and duplicated-window controls, we added a fixed daily seasonal-naive diagnostic for EPİAŞ. This comparison revealed an important limitation: all 15 neural training conditions were worse than the naive baseline. We now report this result explicitly and distinguish comparisons of incremental augmentation benefit from the separate question of absolute predictive competitiveness. We have not tuned the neural architectures or augmentation parameters against the held-out test outcomes and make no claim that the observed distributional differences causally explain the negative result.

## Caveats

The 24-step naive horizon corresponds to 24 hours for EPİAŞ hourly data, but NOT for 10-minute Weather; do not copy this baseline horizon blindly to other domains. This is a post-hoc descriptive adequacy check, not an outcome-driven hyperparameter selection exercise. Do not claim statistical significance from 15 correlated training fits on one fixed test period. Stronger seasonal baseline comparisons can be added to other domains using their true sampling cadence and pre-specified seasonalities.
