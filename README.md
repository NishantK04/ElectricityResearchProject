# Household Electricity Consumption Prediction - Capstone

This repository contains the code and preliminary validation results for the B.Tech capstone project: **"Comparative Study of ML Methods for Household Electricity Consumption Prediction"**.

## Project Overview
This project investigates how to maintain the accuracy of Machine Learning (ML) models predicting household electricity consumption when underlying usage patterns change over time (concept drift). We compare a Static baseline Random Forest against three online adaptation strategies:
1. **Periodic 7-Day Retraining** (Accumulating all history)
2. **Sliding Window 7-Day Retraining** (Recent history only)
3. **ADWIN Drift Detection** (Adaptive windowing based on error streams)

## Key Technical Features
* **Zero Future-Leakage Pipeline:** Custom data preparation ensures that all models are strictly evaluated chronologically. Standard pandas interpolation was explicitly disabled to prevent `t+1` data from leaking into `t` predictions.
* **Fair Telemetry:** The evaluation framework separates `initial_fit`, `retraining`, and `inference` timing to provide a perfectly isolated assessment of algorithmic computational cost vs. accuracy.

## Repository Structure
* `data/raw/` - Contains the raw CEEW smart-meter data.
* `data/processed/` - Contains the `bareilly_ml_features_online_safe_final.csv` used for the experiments.
* `src/` - Python scripts for data preparation and simulation.
  * `07_create_online_safe_data.py` - Generates the leak-free evaluation features.
  * `09_small_timing_adwin_experiment.py` - Runs the core methodology on a small validation dataset.
  * `10_generate_final_figures.py` - Generates publication-ready plots.
* `results/` - Output predictions, timing telemetry, and recalculated metrics CSVs.
* `results/figures/final/` - Generated plots for accuracy and computational cost trade-offs.
* `reports/` - All project documentation, viva questions, abstracts, and audits.

## How to Run (Preliminary Small-Scale Test)
1. Ensure `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `river` are installed.
2. Run `python src/09_small_timing_adwin_experiment.py` to execute the validation framework on the 5-household sample.
3. Run `python src/10_generate_final_figures.py` to regenerate the plots.

*(Note: The full 38-household, 580-day master script is pending final execution and will require substantial computational runtime).*
