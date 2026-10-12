# PROJECT SUMMARY

**Project Title:** Comparative Study of ML Methods for Household Electricity Consumption Prediction

## Overview
This research project investigates how existing machine-learning models behave when household electricity consumption patterns change over time (concept drift). Rather than attempting to invent a novel algorithm, this capstone rigorously compares existing strategies for keeping models updated. The research answers the question: *Which existing ML updating strategy provides the best balance between prediction accuracy and computational cost?*

## Methodology
The project utilizes the CEEW Bareilly smart-meter dataset. A critical technical achievement of the project was identifying and eliminating subtle "future data leakage" caused by standard pandas interpolation functions, resulting in a cryptographically strict chronological walk-forward evaluation pipeline. 

Four approaches were tested to predict the next hour's energy consumption:
1. **Static:** Trained once on historical data; never updated.
2. **Periodic 7-Day:** Retrained every 7 days using all accumulated historical data.
3. **Sliding Window 7-Day:** Retrained every 7 days using only the most recent 7 days of data.
4. **ADWIN:** Retrained only when a statistical drift detector identifies a significant shift in the rolling prediction error.

## Preliminary Findings (Small-Scale Validation)
To validate the methodology before executing a highly expensive full-scale evaluation, a small experiment (5 households, 42 days) was conducted. 

* **Accuracy:** Periodic Retraining consistently outperformed the Static baseline, lowering the Mean Absolute Error (MAE) and capturing shifting consumption trends.
* **Cost:** The computational cost of Periodic Retraining ballooned linearly as history accumulated (taking ~9s to retrain). In contrast, the Sliding Window strategy achieved near-identical accuracy adaptations but completed retrainings in fractions of a second (~0.48s).
* **Drift Detection:** In this specific 42-day micro-sample, ADWIN did not detect statistically significant structural drift, behaving identically to the Static model. 

## Current Status and Future Work
The project's computational framework, telemetry timers, and leakage-prevention mechanisms are now fully audited and scientifically sound. The remaining work involves executing this exact framework on the full 580-day, 38-household dataset to generate the final conclusions for the thesis defense.
