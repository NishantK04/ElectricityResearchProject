import pandas as pd
import numpy as np
import os

processed_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\data\processed"

print("Creating online-safe datasets for 10 households for 90 days...")

df = pd.read_csv(os.path.join(processed_dir, "bareilly_hourly_clean.csv"))
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])

selected_meters = ['BR02', 'BR04', 'BR05', 'BR06', 'BR08', 'BR09', 'BR11', 'BR12', 'BR13', 'BR15']
df = df[df['meter'].isin(selected_meters)].copy()

start_dt = pd.to_datetime("2020-01-01 00:00:00")
end_dt = pd.to_datetime("2020-06-21 23:59:59") # 90 days from March 24
df = df[(df['x_Timestamp'] >= start_dt) & (df['x_Timestamp'] <= end_dt)].copy()

df.sort_values(['meter', 'x_Timestamp'], inplace=True)

df.to_csv(os.path.join(processed_dir, "bareilly_hourly_online_safe_final.csv"), index=False)
print("Saved bareilly_hourly_online_safe_final.csv")

features_list = []
for meter, group in df.groupby('meter'):
    group = group.copy().sort_values('x_Timestamp')
    
    group['target_next_hour'] = group['t_kWh'].shift(-1)
    group = group.dropna(subset=['target_next_hour'])
    
    group['lag_1h'] = group['t_kWh'].shift(1)
    group['lag_2h'] = group['t_kWh'].shift(2)
    group['lag_3h'] = group['t_kWh'].shift(3)
    group['lag_6h'] = group['t_kWh'].shift(6)
    group['lag_12h'] = group['t_kWh'].shift(12)
    group['lag_24h'] = group['t_kWh'].shift(24)
    group['lag_48h'] = group['t_kWh'].shift(48)
    group['lag_168h'] = group['t_kWh'].shift(168)
    
    group['rolling_mean_3h'] = group['t_kWh'].rolling(window=3, min_periods=1).mean()
    group['rolling_mean_6h'] = group['t_kWh'].rolling(window=6, min_periods=1).mean()
    group['rolling_mean_24h'] = group['t_kWh'].rolling(window=24, min_periods=1).mean()
    group['rolling_mean_168h'] = group['t_kWh'].rolling(window=168, min_periods=1).mean()
    group['rolling_std_24h'] = group['t_kWh'].rolling(window=24, min_periods=2).std()
    
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
    
    group.dropna(inplace=True)
    features_list.append(group)

features_df = pd.concat(features_list, ignore_index=True)
features_df.to_csv(os.path.join(processed_dir, "bareilly_ml_features_online_safe_final.csv"), index=False)
print("Saved bareilly_ml_features_online_safe_final.csv")
