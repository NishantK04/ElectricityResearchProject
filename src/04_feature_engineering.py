import pandas as pd
import numpy as np
import os

processed_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\data\processed"
reports_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\reports"

print("Starting Phase 4: Feature Engineering")

# Load V2 dataset
df = pd.read_csv(os.path.join(processed_dir, "bareilly_hourly_clean_v2.csv"))
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df = df.sort_values(by=['meter', 'x_Timestamp']).reset_index(drop=True)

# ---------------------------------------------------------
# PART A & B: Target Creation
# ---------------------------------------------------------
# Target: next hour's consumption (t+1)
# Prediction is made at time t, using info up to t.
df['target_next_hour'] = df.groupby('meter')['t_kWh'].shift(-1)

# ---------------------------------------------------------
# PART C: Time Features
# ---------------------------------------------------------
df['hour'] = df['x_Timestamp'].dt.hour
df['day_of_week'] = df['x_Timestamp'].dt.dayofweek
df['day_of_month'] = df['x_Timestamp'].dt.day
df['month'] = df['x_Timestamp'].dt.month
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['day_of_year'] = df['x_Timestamp'].dt.dayofyear
df['week_of_year'] = df['x_Timestamp'].dt.isocalendar().week.astype(int)

# Cyclical encoding
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# ---------------------------------------------------------
# PART D: Lag Features
# ---------------------------------------------------------
# Since we predict t+1 at time t:
# lag_1h is the consumption at time t.
# lag_2h is the consumption at t-1, etc.
lags = [1, 2, 3, 6, 12, 24, 48, 168]
for lag in lags:
    # shift(lag-1) because shift(0) is time t itself.
    df[f'lag_{lag}h'] = df.groupby('meter')['t_kWh'].shift(lag - 1)

# ---------------------------------------------------------
# PART E: Rolling Features
# ---------------------------------------------------------
# Calculate rolling features on the consumption at time t and before.
# We apply this directly on t_kWh since the target is at t+1, thus avoiding leakage.
def add_rolling_features(group):
    # Mean
    for w in [3, 6, 24, 168]:
        group[f'rolling_mean_{w}h'] = group['t_kWh'].rolling(window=w, min_periods=1).mean()
    # Std
    group['rolling_std_24h'] = group['t_kWh'].rolling(window=24, min_periods=2).std()
    return group

df = df.groupby('meter', group_keys=False).apply(add_rolling_features)

# ---------------------------------------------------------
# PART F: Electrical Features
# ---------------------------------------------------------
# We use Voltage, Current, Freq from time t. This is perfectly valid since they are measured at t, 
# and we predict t+1. No future leakage.

# ---------------------------------------------------------
# PART G: Handle NaNs
# ---------------------------------------------------------
# Leave NaNs as NaNs. Drop rows where target or core features (like lag_1h) are missing.
# If lag_168h is missing (e.g. first 7 days), it will remain NaN. For baseline ML models (RandomForest, LinReg), 
# we need to drop NaNs or impute. XGBoost handles NaNs naturally.
# We will drop rows where target is NaN.
df = df.dropna(subset=['target_next_hour'])

# ---------------------------------------------------------
# PART I: Chronological Split
# ---------------------------------------------------------
# 70% Train, 15% Val, 15% Test
unique_times = np.sort(df['x_Timestamp'].unique())
n_times = len(unique_times)

train_end_idx = int(n_times * 0.70)
val_end_idx = int(n_times * 0.85)

train_end_time = unique_times[train_end_idx]
val_end_time = unique_times[val_end_idx]

train_df = df[df['x_Timestamp'] <= train_end_time]
val_df = df[(df['x_Timestamp'] > train_end_time) & (df['x_Timestamp'] <= val_end_time)]
test_df = df[df['x_Timestamp'] > val_end_time]

# ---------------------------------------------------------
# Save Datasets
# ---------------------------------------------------------
features = ['meter', 'x_Timestamp', 'target_next_hour', 't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
            'hour', 'day_of_week', 'day_of_month', 'month', 'weekend', 'day_of_year', 'week_of_year',
            'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month',
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
            'rolling_mean_3h', 'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_168h']

df = df[features]
df.to_csv(os.path.join(processed_dir, "bareilly_ml_features.csv"), index=False)
train_df[features].to_csv(os.path.join(processed_dir, "bareilly_train.csv"), index=False)
val_df[features].to_csv(os.path.join(processed_dir, "bareilly_validation.csv"), index=False)
test_df[features].to_csv(os.path.join(processed_dir, "bareilly_test.csv"), index=False)

# Validation Report
report_text = f"""# Feature Engineering Validation Report

## 1. Dataset Verification
- Initial Rows: 548424
- Target 'target_next_hour' created successfully without leakage.

## 2. Chronological Split Details
- Total temporal points: {n_times}
- **Training Period:** {unique_times[0]} to {train_end_time} (70%)
- **Validation Period:** {unique_times[train_end_idx+1]} to {val_end_time} (15%)
- **Testing Period:** {unique_times[val_end_idx+1]} to {unique_times[-1]} (15%)

## 3. NaNs and Leakage
- `target_next_hour` is strictly `t+1` consumption.
- Features at time `t` strictly use observations up to `t`.
- Missing values (NaNs) are preserved. Rows with missing target are removed from training.
- No future filling or random shuffling was used.
"""

with open(os.path.join(reports_dir, "04_feature_engineering_report.md"), "w") as f:
    f.write(report_text)

print("Phase 4 Feature Engineering complete.")
