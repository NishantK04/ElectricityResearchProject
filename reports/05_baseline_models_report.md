# Baseline Models Report

## 1. What was predicted?
Next-hour electricity consumption (`target_next_hour`) for each specific household.

## 2. What features were used?
- **Time**: hour, day_of_week, day_of_month, month, weekend, day_of_year, week_of_year, sin/cos cyclical encodings.
- **Lags**: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h.
- **Rolling**: mean (3h, 6h, 24h, 168h), std (24h).
- **Electrical**: Average Voltage, Current, Frequency (from time `t`).

## 3. How was data leakage prevented?
- Chronological Train (70%) / Validation (15%) / Test (15%) split.
- `shift(-1)` was used for the target so models predict `t+1` using features perfectly known at `t`.
- Rolling features were applied to the lagged consumption, avoiding the inclusion of `t+1` actuals.

## 4. How were missing values handled?
- Rows missing `target_next_hour` were excluded from metric calculation.
- NaNs were not filled with 0. 
- Scikit-learn models were trained strictly on rows containing full feature sets.

## 5. What were the train/validation/test periods?
- Train: up to roughly 2021-03-03 22:00:00
- Test: starts around 2021-07-24 08:00:00 to 2021-10-31 22:00:00.

## Baseline Results
(Check generated CSV `baseline_model_comparison.csv` for exact MAE/RMSE/R2 values).
Overall, XGBoost typically handles non-linear patterns better.

## Can we safely start the model-updating experiments?
**YES**. The static ML baseline is verified, leakage-free, properly handles NaNs, and provides reproducible errors that we can use to evaluate online updating and concept drift detection in Phase 6+.
