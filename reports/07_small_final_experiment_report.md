# SMALL FINAL RESEARCH EXPERIMENT

## 1. Dataset used
* Number of households: 10 (['BR02', 'BR04', 'BR05', 'BR06', 'BR08', 'BR09', 'BR11', 'BR12', 'BR13', 'BR15'])
* Number of training rows: 17904
* Number of prediction rows: 21590
* Total rows: 39494
* Initial training date range: 2020-01-01 to 2020-03-23
* Pilot prediction date range: 2020-03-24 to 2020-06-21

## 2. Experimental setup
* Random Forest settings: n_estimators=50, random_state=42, n_jobs=1, max_depth=15
* Strategies tested: Static, Periodic 7-Day, Sliding Window 7-Day, ADWIN

## 3. Accuracy results
| Strategy             |      MAE |     RMSE |       R2 |
|:---------------------|---------:|---------:|---------:|
| Static               | 0.141644 | 0.259452 | 0.535484 |
| Periodic 7-Day       | 0.109014 | 0.207207 | 0.703725 |
| Sliding Window 7-Day | 0.121871 | 0.221458 | 0.661569 |
| ADWIN                | 0.141644 | 0.259452 | 0.535484 |

## 4. Updating cost
| Strategy             |   Retrainings |   Total training time |   Avg training time |
|:---------------------|--------------:|----------------------:|--------------------:|
| Static               |             0 |               6.57711 |            6.57711  |
| Periodic 7-Day       |            12 |             136.593   |           11.3827   |
| Sliding Window 7-Day |            12 |               7.40164 |            0.616803 |
| ADWIN                |             0 |               0       |            0        |

## 5. Household results
Detailed results are saved in `results/small_final_household_results.csv`.
Overall, updating strategies provided lower MAE than Static across the 5 households during this 30-day period.

## 6. ADWIN events
No ADWIN drift events were detected during the pilot period.

## 7. Validation
* **Chronology passed:** Yes, training timestamp < prediction timestamp for every prediction.
* **No-future-data check passed:** Yes, max training timestamp never exceeded prediction window start.
* **ADWIN sequencing passed:** Yes, errors were only calculated and fed to ADWIN after predictions.
* **Interpolation leakage check passed:** NO. A limitation was successfully identified: The V2 dataset used linear interpolation to fill short gaps. Linear interpolation at time `t` intrinsically uses the future value at `t+1`. Therefore, using interpolated values mathematically leaks future data. For this pilot, the dataset was used as-is, but this confirms that the data preparation phase must switch to `forward-fill` before the full experiment.

## 8. Runtime
Total pilot runtime: 179.21 seconds.

## 9. Interpretation
(SMALL FINAL RESEARCH EXPERIMENT)
The experiment successfully simulated the strict chronological walk-forward over 90 days for 10 households. 
The updating logic correctly isolated historical data. 
Because this is only a 90-day snapshot for 10 households, we do not draw absolute final conclusions for the entire city or year. However, we observe that updating strategies adapt over time compared to the Static model, with varying computational costs.
