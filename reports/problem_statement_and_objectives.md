# PROBLEM STATEMENT AND OBJECTIVES

## Problem Statement
Accurate household-level electricity consumption forecasting is essential for smart grid optimization and demand-side management. However, household consumption is non-stationary; patterns change abruptly or gradually due to seasonality, behavioral shifts, or external events (concept drift). Traditional machine learning models are typically trained once offline and deployed statically. Over time, as the household's actual consumption patterns diverge from the historical training data, the static model's predictive accuracy degrades. While continuous retraining can mitigate this degradation, retraining on all accumulated historical data imposes severe computational costs, particularly if scaled to thousands of smart meters. There is a need to systematically compare existing model-updating strategies to identify which approach provides the optimal balance between predictive accuracy and computational cost in the context of household electricity forecasting.

## Objectives
1. To engineer a strictly chronological, leak-free data processing and evaluation pipeline that prevents future information from artificially inflating prediction accuracy.
2. To implement and evaluate a Static baseline model (Random Forest) against three existing online adaptation strategies: Periodic Retraining, Sliding Window Retraining, and ADWIN-based active drift detection.
3. To empirically quantify the trade-off between prediction accuracy (MAE, RMSE, R²) and computational cost (retraining time) across the implemented strategies using real-world smart meter data.
4. To establish a robust preliminary methodology on a subset of households that can be securely scaled to evaluate a multi-year, city-wide dataset.
