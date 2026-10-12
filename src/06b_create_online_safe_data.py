import pandas as pd
import numpy as np
import os

processed_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\data\processed"

print("Creating online-safe datasets for 5 households...")

# Load the un-interpolated dataset
df = pd.read_csv(os.path.join(processed_dir, "bareilly_hourly_clean.csv"))
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])

# Keep only 5 households
selected_meters = ['BR02', 'BR04', 'BR05', 'BR06', 'BR08']
df = df[df['meter'].isin(selected_meters)].copy()

# Keep only the required dates: Jan 1, 2020 - April 22, 2020
start_dt = pd.to_datetime("2020-01-01 00:00:00")
end_dt = pd.to_datetime("2020-04-22 23:59:59")
df = df[(df['x_Timestamp'] >= start_dt) & (df['x_Timestamp'] <= end_dt)].copy()

df.sort_values(['meter', 'x_Timestamp'], inplace=True)

# Save the online-safe hourly base dataset (no linear interpolation)
df.to_csv(os.path.join(processed_dir, "bareilly_hourly_online_safe_pilot.csv"), index=False)
print("Saved bareilly_hourly_online_safe_pilot.csv")

# Now engineer features ONLINE-SAFE
# No future data leaks! 

features_list = []
for meter, group in df.groupby('meter'):
    group = group.copy().sort_values('x_Timestamp')
    
    # Target: next hour's actual value
    # Shift(-1) pulls t+1 into t. This is mathematically correct for training/predicting.
    group['target_next_hour'] = group['t_kWh'].shift(-1)
    
    # Drop rows where target is NaN so we don't evaluate on missing targets
    group = group.dropna(subset=['target_next_hour'])
    
    # Lags MUST only use past/current information.
    # At time t, we know t_kWh (which is consumption during the hour ending at t).
    # So lag_1h is just t_kWh from 1 hour ago.
    group['lag_1h'] = group['t_kWh'].shift(1)
    group['lag_2h'] = group['t_kWh'].shift(2)
    group['lag_3h'] = group['t_kWh'].shift(3)
    group['lag_6h'] = group['t_kWh'].shift(6)
    group['lag_12h'] = group['t_kWh'].shift(12)
    group['lag_24h'] = group['t_kWh'].shift(24)
    group['lag_48h'] = group['t_kWh'].shift(48)
    group['lag_168h'] = group['t_kWh'].shift(168)
    
    # Rolling features MUST use historical values, INCLUDING the current hour's value which is known.
    # Pandas rolling(window=3) includes the current row and the 2 previous rows.
    # This is safe because at time t, we know t, t-1, t-2. 
    group['rolling_mean_3h'] = group['t_kWh'].rolling(window=3, min_periods=1).mean()
    group['rolling_mean_6h'] = group['t_kWh'].rolling(window=6, min_periods=1).mean()
    group['rolling_mean_24h'] = group['t_kWh'].rolling(window=24, min_periods=1).mean()
    group['rolling_mean_168h'] = group['t_kWh'].rolling(window=168, min_periods=1).mean()
    group['rolling_std_24h'] = group['t_kWh'].rolling(window=24, min_periods=2).std()
    
    # Time features (only derived from current timestamp)
    group['hour'] = group['x_Timestamp'].dt.hour
    group['day_of_week'] = group['x_Timestamp'].dt.dayofweek
    group['month'] = group['x_Timestamp'].dt.month
    group['is_weekend'] = group['day_of_week'].isin([5, 6]).astype(int)
    
    group['hour_sin'] = np.sin(2 * np.pi * group['hour'] / 24)
    group['hour_cos'] = np.cos(2 * np.pi * group['hour'] / 24)
    group['day_sin'] = np.sin(2 * np.pi * group['day_of_week'] / 7)
    group['day_cos'] = np.cos(2 * np.pi * group['day_of_week'] / 7)
    group['month_sin'] = np.sin(2 * np.pi * group['month'] / 12)
    group['month_cos'] = np.cos(2 * np.pi * group['month'] / 12)
    
    # No filling of NaNs with forward values. 
    # Just drop rows that still have NaNs in features.
    group.dropna(inplace=True)
    
    # Electrical features are available at time t
    # 'Avg Voltage', 'Avg Current', 'Freq'
    
    features_list.append(group)

features_df = pd.concat(features_list, ignore_index=True)
features_df.rename(columns={'x_Timestamp': 'x_Timestamp'}, inplace=True)
features_df.to_csv(os.path.join(processed_dir, "bareilly_ml_features_online_safe_pilot.csv"), index=False)
print("Saved bareilly_ml_features_online_safe_pilot.csv")
