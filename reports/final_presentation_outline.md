# FINAL PRESENTATION OUTLINE

**Target Duration:** 15 Minutes
**Total Slides:** 14

## Slide 1: Title Slide
* **Title:** Comparative Study of ML Methods for Household Electricity Consumption Prediction
* **Student Name / ID**
* **Guide Name**

## Slide 2: Introduction & Motivation
* The rise of smart meters and smart grids.
* Why household-level forecasting is important (demand response, grid stability).
* The problem: Households are not factories. Their electricity usage is volatile and changes over time (seasons, holidays, new appliances, routines).

## Slide 3: The Problem of Concept Drift
* Define Concept Drift simply: When the statistical properties of the target variable change over time.
* Explain the flaw in traditional ML: A model trained on Winter data will lose accuracy in Summer.
* The Research Gap: We know models decay, but what is the most efficient existing way to keep them updated at the edge (household level)?

## Slide 4: Research Objectives
1. Build a strict, leak-free evaluation pipeline that mathematically prevents future-data cheating.
2. Compare a Static model against 3 updating strategies (Periodic, Sliding Window, ADWIN).
3. Quantify the exact trade-off between Prediction Accuracy and Computational Retraining Cost.

## Slide 5: Dataset & Preprocessing
* **Source:** CEEW Bareilly Smart Meter Dataset.
* **Granularity:** Converted from raw arbitrary intervals to structured Hourly predictions.
* **Missing Data:** Addressed how missing values were handled using forward-filling to avoid the future-leakage caused by linear interpolation.

## Slide 6: Feature Engineering
* Goal: Predict the *next* hour's consumption ($t+1$).
* Features used:
    * Temporal features (Hour of day, Day of week).
    * Historical lags ($t$, $t-1$, $t-24$).
    * Rolling statistics (Mean over last 3 hours, 24 hours).
* Emphasize: Every feature represents information strictly available at time $t$.

## Slide 7: The Challenge of "Future Leakage"
* Explain the critical bug discovered during Phase 6.
* Using Pandas `interpolate()` mathematically uses $t+1$ to guess $t$. 
* Show how the Walk-Forward validation chronologically isolates the training set from the test set to guarantee scientific integrity.

## Slide 8: The 4 Strategies Compared
1. **Static:** Train once, deploy forever. (Baseline)
2. **Periodic (Accumulating):** Retrain every 7 days on *all* history.
3. **Sliding Window:** Retrain every 7 days on *only* the last 7 days.
4. **ADWIN (Adaptive Windowing):** Retrain only when a statistical algorithm detects that the error rate has shifted significantly.

## Slide 9: Preliminary Results - Accuracy
* *(Include figure: `final_mae_comparison.png`)*
* Small-scale validation results (5 households, 42 days).
* Show that Periodic Retraining successfully lowered the MAE compared to the Static model.

## Slide 10: Preliminary Results - Computational Cost
* *(Include figure: `final_accuracy_vs_cost.png`)*
* Show the timing comparisons.
* Explain that Periodic retraining took ~9 seconds because data constantly grows.
* Explain that Sliding Window took ~0.48 seconds with competitive accuracy.

## Slide 11: ADWIN Behavior
* Explain how ADWIN was tested on a validation subset.
* Share that in this specific 42-day window, the variance was too stable, so ADWIN correctly decided *not* to trigger a retraining.
* It safely defaulted to behaving like the Static model.

## Slide 12: Discussion & Trade-offs
* There is no "perfect" algorithm.
* If a utility company has massive cloud servers, Periodic (accumulating) yields the best accuracy.
* If the model must run locally on a cheap smart meter (edge computing), Sliding Window is vastly superior due to near-zero retraining time.

## Slide 13: Limitations
* The results shown are from a small 42-day micro-sample used to validate the pipeline.
* Did not include external weather data (temperature/humidity).
* Cannot conclusively claim sliding-window is universally better without testing a full annual cycle.

## Slide 14: Conclusion & Future Work
* Built a fully automated, strictly leak-free framework for time-series evaluation.
* Proved the computational trade-offs of online updating.
* **Next step:** Execute this exact framework on the full 38-household, 580-day master dataset for the final publication.
* **Thank You / Q&A.**
