# Hourly Dataset Validation Report

## 1. Verifying t_kWh
- Monotonically increasing for BR02: False
- Median value: 0.009
- **Conclusion**: `t_kWh` represents interval energy consumption (energy consumed during the 3-minute interval), NOT a cumulative meter reading.
- **Aggregation Method**: Summing `t_kWh` to create hourly consumption is correct.

## 2. Missing Hourly Data Investigation
- Total Expected Hourly Records: 548424
- Total Observed Hourly Records: 490776
- Total Missing Hourly Records: 57648
- Percentage of Missing Hourly Records: 10.51%

## 3. Zero Imputation Validity
Replacing missing electricity consumption with `0` is **NOT scientifically justified**. A missing hour means the meter lost communication or was turned off, not necessarily that household consumption was strictly 0.00 kWh. Using 0 creates artificial drops.
Decision: We will keep missing values as NaN (Option B) for consumption, or perform interpolation for short gaps (Option C) and drop long gaps.

## 4. False Concept Drift Check
If we kept zeros, artificial drops would definitely trigger false drift alarms (e.g., consumption going from 1.5 kWh -> 0.0 -> 1.2 kWh). Leaving them as NaN or intelligently interpolating prevents this.

## 7. Rebuild Decision
A. Is t_kWh aggregation correct? YES (Sum)
B. Is missing consumption = 0 scientifically justified? NO
C. Should the hourly dataset be regenerated? YES
- Regenerated dataset saved as `bareilly_hourly_clean_v2.csv`.
- Changes: Used linear interpolation for gaps up to 2 hours. Left longer gaps as NaN to avoid false concept drift.
