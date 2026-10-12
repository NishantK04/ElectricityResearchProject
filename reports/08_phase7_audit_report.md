# PHASE 7 AUDIT REPORT

## 1. Methodology Verification

An audit of `src/07_small_final_experiment.py` confirmed the following methodological checks:
1. **Information availability:** `PASS`. Predictions strictly use features generated from historical data. The walk-forward loop strictly uses `x_Timestamp <= current_time` for training and `x_Timestamp > current_time` for prediction.
2. **Next-hour target:** `PASS`. The target variable `target_next_hour` is explicitly generated as `t_kWh.shift(-1)`.
3. **No future training targets:** `PASS`. Because training data is strictly capped at `x_Timestamp <= current_time`, the model only trains on electricity consumption that has already finished.
4. **Periodic retraining data:** `PASS`. Uses `df[x_Timestamp <= current_time]`, accumulating all eligible history.
5. **Sliding-window retraining data:** `PASS`. Uses a strictly bounded window of `current_time - 7 days` to `current_time`.
6. **ADWIN sequencing:** `PASS`. The simulation generates predictions for a chunk, calculates the actual `daily_mae` for those predictions using the known targets, and then feeds that error to ADWIN, simulating end-of-day error ingestion. 
7. **Consistent evaluation set:** `PASS`. All four strategies output exactly 21,590 predictions over the same timestamps and households.
8. **Missing observations:** `PASS`. The online-safe data generation (`src/07_create_online_safe_data.py`) explicitly drops missing observations rather than interpolating them or filling them with zeros.
9. **Metrics calculation:** `PASS`. Metrics independently re-calculated from `results/small_final_predictions.csv` perfectly match the saved summary CSV.
10. **Retraining counts:** `PASS`. 12 retrainings for the 7-day intervals over the 90-day period perfectly aligns with expectations.

## 2. Audit of Training Cost and Timing

The reported training times in Phase 7 were severely skewed due to inconsistent timestamping in the code:
* **Static:** Reported 6.57s. However, its timing logic (`t0 = time.time()`) started before model fitting and ended after prediction and DataFrame manipulations.
* **Periodic & Sliding Window:** Reported 136.59s and 7.40s. Their timing logic accurately accumulated only the *subsequent retraining times* (`tot_train_t += dur`), completely ignoring the initial model fit.
* **ADWIN:** Reported 0s. Like Periodic, it ignored the initial model fit, and because it never triggered a drift retraining, its accumulated retraining time was literally 0.

**Conclusion:** The total training time metric is currently unfair because Static includes inference and DataFrame overhead, while the other strategies only measure retraining time and exclude their initial fit. This discrepancy must be fixed before final publication. 

## 3. Recalculated Results

Using `results/small_final_predictions.csv`, the metrics were independently recalculated across all 21,590 test observations:

| Strategy | MAE | RMSE | R2 | N |
| :--- | :--- | :--- | :--- | :--- |
| **Static** | 0.141644 | 0.259452 | 0.535484 | 21590 |
| **Periodic 7-Day** | 0.109014 | 0.207207 | 0.703725 | 21590 |
| **Sliding Window 7-Day** | 0.121871 | 0.221458 | 0.661569 | 21590 |
| **ADWIN** | 0.141644 | 0.259452 | 0.535484 | 21590 |

The recalculated metrics exactly match the originally reported results.

## 4. Assessment of Conclusions

* **Periodic retraining outperformed the static model:** **Supported.** Periodic 7-Day significantly dropped MAE from 0.141 to 0.109 and increased $R^2$ from 0.535 to 0.703 over a consistent, isolated evaluation set.
* **Sliding-window retraining used less training time:** **Supported.** Sliding window only accumulated 7.40s of retraining time compared to Periodic's 136.59s, due to strictly training on 7-day chunks rather than accumulating 90+ days of history.
* **ADWIN detected no drift events:** **Supported.** The ADWIN MAE precisely mirrors the Static MAE, indicating 0 retrainings. The `delta=0.002` threshold was not breached by the daily MAE error signal in this 10-household sample.

## 5. Next Steps for Final Research
Phase 7 serves as an excellent, fully leak-free preliminary research evidence. Before the full thesis run, the following must be fixed:
1. **Timing Logic:** The timing code must be standardized so that *Initial Fit Time*, *Retraining Time*, and *Inference Time* are distinct, isolated variables for a fair comparison.
2. **ADWIN Tuning:** Since ADWIN failed to trigger, its sensitivity `delta` (or the daily aggregation method) needs to be tuned on a validation set so it isn't completely dormant.
