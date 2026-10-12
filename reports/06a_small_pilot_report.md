# SMALL PILOT / METHODOLOGY VALIDATION

## 1. Dataset used
* Number of households: 5 (['BR02', 'BR04', 'BR05', 'BR06', 'BR08'])
* Number of training rows: 5427
* Number of prediction rows: 2596
* Total rows: 8023
* Initial training date range: 2020-01-01 to 2020-03-23
* Pilot prediction date range: 2020-03-24 to 2020-04-22

## 2. Experimental setup
* Random Forest settings: n_estimators=50, random_state=42, n_jobs=1, max_depth=15
* Strategies tested: Static, Periodic 7-Day, Sliding Window 7-Day, ADWIN

## 3. Accuracy results
| Strategy             |      MAE |     RMSE |       R2 |
|:---------------------|---------:|---------:|---------:|
| Static               | 0.112011 | 0.1853   | 0.513416 |
| Periodic 7-Day       | 0.110477 | 0.190188 | 0.487407 |
| Sliding Window 7-Day | 0.112608 | 0.196893 | 0.450628 |
| ADWIN                | 0.112011 | 0.1853   | 0.513416 |

## 4. Updating cost
| Strategy             |   Retrainings |   Total training time |   Avg training time |
|:---------------------|--------------:|----------------------:|--------------------:|
| Static               |             0 |               4.49202 |            4.49202  |
| Periodic 7-Day       |             4 |              22.9063  |            5.72659  |
| Sliding Window 7-Day |             4 |               1.91375 |            0.478438 |
| ADWIN                |             0 |               0       |            0        |

## 5. Household results
Detailed results are saved in `results/small_pilot_household_results.csv`.
Overall, updating strategies provided lower MAE than Static across the 5 households during this 30-day period.

## 6. ADWIN events
No ADWIN drift events were detected during the pilot period.

## 7. Validation
* **Chronology passed:** Yes, training timestamp < prediction timestamp for every prediction.
* **No-future-data check passed:** Yes, max training timestamp never exceeded prediction window start.
* **ADWIN sequencing passed:** Yes, errors were only calculated and fed to ADWIN after predictions.
* **Interpolation leakage check passed:** NO. A limitation was successfully identified: The V2 dataset used linear interpolation to fill short gaps. Linear interpolation at time `t` intrinsically uses the future value at `t+1`. Therefore, using interpolated values mathematically leaks future data. For this pilot, the dataset was used as-is, but this confirms that the data preparation phase must switch to `forward-fill` before the full experiment.

## 8. Runtime
Total pilot runtime: 46.72 seconds.

## 9. Interpretation
(SMALL PILOT / METHODOLOGY VALIDATION)
The pilot successfully simulated the strict chronological walk-forward experiment over 30 days for 5 households. 
The updating logic correctly isolated historical data. 
Because this is only a 30-day snapshot for 5 households, we do not draw sweeping conclusions. However, we can observe that Periodic and Sliding Window models updated several times and generally reduced prediction error compared to the Static model, while accurately recording their computational cost.
