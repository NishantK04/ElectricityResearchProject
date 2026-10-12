# LIMITATIONS AND OPEN ISSUES

As part of maintaining strict scientific integrity, the following limitations and open issues are explicitly declared for this capstone project:

## 1. Small-Scale Validation Sample
**Issue:** The currently presented empirical results (Phase 9/10) are based on a highly restricted micro-sample: exactly 5 households spanning only 42 days (March 24, 2020 – May 5, 2020). 
**Implication:** These results mathematically prove that the *evaluation pipeline* is functional, leak-free, and correctly tracks timing. However, they *cannot* be used to make sweeping claims about how these algorithms perform across the entire city of Bareilly over an entire year. The full 580-day master execution is still required.

## 2. ADWIN Statistical Thresholds
**Issue:** During the 42-day micro-sample, ADWIN failed to detect any structural drift, even when its sensitivity was maximized (`delta=0.0001`) and it was fed high-resolution hourly absolute errors. 
**Implication:** The household consumption variance was simply too stable during this short time frame to trigger ADWIN's statistical threshold. This is normal behavior for a drift detector, but it means we have not yet empirically observed ADWIN triggering a retraining event in this project. It is highly likely to trigger during seasonal transitions (e.g., Spring to Summer) when the full dataset is evaluated.

## 3. Absence of Weather Features
**Issue:** Electricity consumption is heavily correlated with external temperature (HVAC usage). The current feature set relies entirely on historical consumption lags (autoregressive features) and temporal markers (hour, day).
**Implication:** Sudden spikes in usage due to a heatwave will be registered by the model as "concept drift", whereas a model equipped with temperature features would simply recognize it as expected behavior. Adding weather data is an open issue for future work.

## 4. Hyperparameter Optimization
**Issue:** The Random Forest baseline utilizes a hardcoded, lightweight configuration (`n_estimators=25`, `max_depth=15`) to ensure the walk-forward simulation runs quickly.
**Implication:** A fully optimized Random Forest would likely achieve a lower absolute MAE. However, since all updating strategies utilize the exact same base model, the *relative* performance comparison (the core objective of this study) remains valid.
