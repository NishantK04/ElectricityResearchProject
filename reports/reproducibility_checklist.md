# REPRODUCIBILITY CHECKLIST

This document ensures that the experiments conducted in this capstone can be independently verified and reproduced by other researchers.

## 1. Environment & Dependencies
- [x] **Python Version:** 3.10+
- [x] **Libraries used:** 
  - `pandas` (Data manipulation)
  - `numpy` (Numerical operations)
  - `scikit-learn` (Random Forest, Metrics)
  - `river` (ADWIN concept drift detection)
  - `matplotlib`, `seaborn` (Visualization)
- [x] **Random Seeds:** `random_state=42` is explicitly enforced in all `RandomForestRegressor` instances.

## 2. Data Provenance
- [x] **Raw Data Access:** The dataset is derived from the CEEW Bareilly smart meter dataset.
- [x] **Data Preprocessing Script:** The exact steps to convert raw data to hourly ML features without future leakage are documented and executable via `src/07_create_online_safe_data.py`.
- [x] **Missing Data Handling:** Explicitly documented as `dropna()` and forward-filling; linear interpolation is banned to prevent leakage.

## 3. Experimental Execution
- [x] **Execution Script:** `src/09_small_timing_adwin_experiment.py`
- [x] **Chronological Splitting:** The code hardcodes exact timestamp barriers (`x_Timestamp <= val_init_train_end`) rather than relying on randomized `train_test_split`.
- [x] **Hyperparameters:**
  - Random Forest: `n_estimators=25`, `max_depth=15`, `n_jobs=1` (forced to 1 to prevent Windows threading deadlocks during rapid retraining loops).
  - ADWIN: `delta=0.001` (chosen via the isolated Phase 9 validation segment).

## 4. Result Verification
- [x] **Metric Recalculation:** `src/10_generate_final_figures.py` and the Phase 10 audit verify that all reported metrics (MAE, RMSE, R²) can be independently recalculated from the raw output `phase9_predictions.csv`.
- [x] **Timing Telemetry:** Timers are isolated strictly to `.fit()` and `.predict()` functions.
