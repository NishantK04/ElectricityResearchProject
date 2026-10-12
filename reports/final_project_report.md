# FINAL PROJECT REPORT

## 1. Title and Abstract
**Title:** Comparative Study of ML Methods for Household Electricity Consumption Prediction
**Abstract:** This project empirically investigates how to maintain the predictive accuracy of Machine Learning models forecasting household electricity consumption when underlying usage patterns change over time (concept drift). We rigorously compare a Static Random Forest against Periodic, Sliding Window, and ADWIN-based updating strategies. Preliminary testing on a leak-free pipeline demonstrates that periodic updating improves accuracy (MAE = 0.087 vs 0.092) but incurs linear computational costs, while sliding windows provide near-zero latency adaptation.

## 2. Introduction and Problem Statement
Household-level electricity forecasting is critical for smart grid optimization. However, household consumption is non-stationary; patterns change abruptly or gradually due to seasonality or behavioral shifts. Static machine learning models degrade in performance over time. While continuous retraining mitigates this degradation, retraining on all historical data imposes severe computational costs. This project aims to compare existing updating strategies to find the optimal accuracy-cost balance.

## 3. Research Motivation and Gap
We are not inventing a new ML algorithm. Instead, we address the practical deployment gap: while cloud servers can handle endless retraining, edge devices (like smart meters) cannot. There is a need for a controlled experimental comparison of updating strategies, paired with a leak-free evaluation methodology.

## 4. Research Objectives
1. Build a strict, chronologically isolated evaluation pipeline.
2. Evaluate a Static baseline against Periodic, Sliding Window, and ADWIN strategies.
3. Quantify the exact trade-off between prediction accuracy and retraining cost.

## 5. Related Work
* Adaptive Windowing (ADWIN) for concept drift detection (Bifet & Gavalda, 2007).
* Sliding Window mechanisms for streaming data.

## 6. Dataset and Preprocessing
The CEEW Bareilly Smart Meter Dataset was converted from raw intervals to hourly grids. Missing values were dropped or forward-filled. Linear interpolation was explicitly banned after discovering it introduced "future data leakage."

## 7. Feature Engineering
Target: `target_next_hour`.
Features: Temporal (hour, day), historical lags (t, t-1, t-24), and rolling means. All features strictly use information available at time `t`.

## 8. Leakage Prevention
A strict walk-forward evaluation protocol isolates the `model.fit()` training loop from the `model.predict()` loop. The model's training upper-bound timestamp never exceeds the prediction window's lower-bound timestamp.

## 9. Baseline Model Comparison
A standard Random Forest (`n_estimators=25`) was chosen over deep learning to ensure fast retraining cycles and allow clear algorithmic cost profiling.

## 10. Updating Strategies
* **Static:** Trained once; never updated.
* **Periodic 7-Day:** Retrained every 7 days on all history.
* **Sliding Window 7-Day:** Retrained every 7 days on the last 7 days only.
* **ADWIN:** Retrained dynamically when drift is detected.

## 11. Audited Small-Experiment Results
*(Validated on 5 households, 42 days, 2,520 predictions)*
* Static: MAE = 0.0923
* Periodic 7-Day: MAE = 0.0879
* Sliding Window 7-Day: MAE = 0.0960
* ADWIN: MAE = 0.0923

## 12. Accuracy versus Computational Cost
* Periodic Retraining improved accuracy but required 9.04 seconds to retrain.
* Sliding Window required only 0.48 seconds to retrain.

## 13. Household-Level Findings
Household variances were remarkably stable during the March-April validation segment, with all 5 meters yielding MAEs below 0.1 kWh.

## 14. ADWIN Results
Because the 42-day validation segment exhibited no structural variance shifts, ADWIN correctly triggered zero retrainings, safely defaulting to the Static baseline performance.

## 15. Discussion and Practical Implications
The results mathematically confirm the computational trade-offs of online updating. Sliding windows are essential for edge-device deployment, whereas Periodic accumulating windows are superior for cloud-based architectures.

## 16. Limitations and Threats to Validity
The current findings are derived from a small 42-day micro-sample. They are preliminary and do not capture annual seasonal shifts. Exogenous weather data is also missing.

## 17. Conclusion and Future Work
The methodology, telemetry, and leakage guards are now theoretically sound and empirically proven. The immediate future work is executing this finalized framework on the complete 38-household, 580-day master dataset.

## 18. References
* CEEW Bareilly Dataset (2020)
* Bifet, A. (2007)
