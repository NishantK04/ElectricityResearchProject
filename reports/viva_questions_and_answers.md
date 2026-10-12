# VIVA QUESTIONS AND ANSWERS

**Q1: What is the main objective of your project?**
**A:** The main objective is to empirically compare different machine learning model-updating strategies (like Periodic retraining vs Sliding Windows vs ADWIN) to see which one best handles changing household electricity consumption patterns (concept drift) without requiring too much computational power.

**Q2: Did you invent a new Machine Learning algorithm?**
**A:** No. The project focuses on applying and rigorously evaluating *existing* adaptation strategies. The contribution is the controlled experimental comparison, the prevention of future-data leakage, and the specific analysis of the accuracy-versus-cost trade-off on smart meter data.

**Q3: What is "Concept Drift"?**
**A:** Concept drift occurs when the statistical properties of the target variable (electricity consumption) change over time. A model trained on past data becomes less accurate because the "concept" of what normal usage looks like has shifted.

**Q4: Why did you use Random Forest instead of Deep Learning (LSTMs, etc.)?**
**A:** Deep learning models take a very long time to train and are highly complex to retrain on the fly. Since our focus was evaluating the *updating strategies* (when and how much data to retrain on) and computational cost, Random Forest provided a strong, fast, traditional ML baseline.

**Q5: What dataset did you use?**
**A:** We used the CEEW smart-meter dataset from Bareilly, India. It contains high-resolution consumption data for various households.

**Q6: What exactly is your model predicting?**
**A:** The model predicts the total electricity consumption (in kWh) for the *next hour*, using only information available up to the current hour.

**Q7: You mentioned "future data leakage." What is that and how did you fix it?**
**A:** Future leakage happens when a model accidentally uses information during training that it wouldn't actually have in real life. During Phase 6, we discovered that Pandas `interpolate(method='linear')` was secretly using future values to fill missing gaps. We fixed it by completely removing interpolation and strictly using `dropna()` and chronologically splitting the data.

**Q8: Explain the difference between Periodic Retraining and Sliding Window.**
**A:** Periodic Retraining takes *all* historical data accumulated from day 1 and retrains the model. Sliding Window only takes the most recent chunk of data (e.g., the last 7 days) and retrains the model, throwing away the older data. 

**Q9: Which strategy had the best accuracy?**
**A:** In our preliminary validation test, Periodic Retraining had the best accuracy (lowest MAE) because it had access to the most data.

**Q10: Which strategy had the best computational cost?**
**A:** The Sliding Window strategy. Because it only trains on the last 7 days, its retraining time was virtually zero (~0.48 seconds), whereas Periodic retraining time keeps growing every week (taking ~9 seconds in the small test).

**Q11: What is ADWIN?**
**A:** ADWIN stands for Adaptive Windowing. It is a statistical algorithm that monitors a stream of numbers (in our case, the model's hourly prediction errors). If the mean of the errors shifts significantly, it triggers an alarm, telling the model to retrain.

**Q12: Did ADWIN detect drift in your experiment?**
**A:** In the small 42-day validation sample, ADWIN did *not* detect drift. The error variance was too stable over that short period. It effectively behaved exactly like the Static baseline model.

**Q13: How did you measure computational cost fairly?**
**A:** We wrote custom telemetry timers that explicitly wrapped *only* the `model.fit()` and `model.predict()` functions. We made sure to separate the "Initial Training Time" from the "Retraining Time" so that the Static model wasn't unfairly penalized by dataframe manipulations.

**Q14: Why didn't you run the full 580-day experiment on all 38 households yet?**
**A:** Running the full experiment is highly computationally expensive. Standard research practice dictates that we first build, debug, and validate the pipeline on a small subset (5 households, 42 days) to ensure the methodology and leakage prevention are flawless before scaling up.

**Q15: How did you handle missing data without leaking the future?**
**A:** If an hour was missing, we dropped it from the training set rather than trying to guess it using future bounds. 

**Q16: What features did your model use?**
**A:** Temporal features (hour of day, day of week) to capture human routines, and historical consumption lags (the usage 1 hour ago, 2 hours ago, and 24 hours ago), plus rolling averages of recent usage.

**Q17: Is it possible that the Periodic model overfits because it trains on so much data?**
**A:** Random Forests are generally robust to overfitting, but yes, older data might become irrelevant. That's exactly why we compare it to the Sliding Window, which aggressively forgets old, potentially irrelevant data.

**Q18: What are the limitations of your project?**
**A:** We did not include weather data (temperature), which heavily drives electricity usage (ACs/heaters). Also, our preliminary results are on a small timeframe, so we can't make absolute claims about the whole year yet.

**Q19: If you were a utility company, which model would you deploy?**
**A:** If computing power is unlimited in the cloud, I'd use Periodic Retraining for maximum accuracy. If the model needs to run locally on a smart meter inside a house (edge computing), I would use Sliding Window because it requires almost no CPU time to update.

**Q20: What is the very next step for this research?**
**A:** To take this fully audited, leak-free codebase and run it on the complete 38-household dataset to generate the final graphs for the thesis publication.
