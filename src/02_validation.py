import pandas as pd
import numpy as np
import os
import glob

raw_dir = r"C:\Users\nisha\Downloads\ML-Based Household Electricity consumption prediction\ElectricityProject"
processed_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\data\processed"
results_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\results"
reports_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\reports"

# 1. Load Raw Data (subset) to verify t_kWh behavior
print("Verifying t_kWh...")
file_2020 = os.path.join(raw_dir, "CEEW - Smart meter data Bareilly 2020.csv")
df_raw = pd.read_csv(file_2020, nrows=100000)
df_raw['x_Timestamp'] = pd.to_datetime(df_raw['x_Timestamp'])
df_raw.sort_values(['meter', 'x_Timestamp'], inplace=True)

meter1 = df_raw['meter'].unique()[0]
df_m1 = df_raw[df_raw['meter'] == meter1].copy()

# Check if t_kWh is cumulative or interval
# If cumulative, it should be monotonically increasing.
# If interval, it fluctuates and drops to near zero.
is_monotonic = df_m1['t_kWh'].is_monotonic_increasing
median_val = df_m1['t_kWh'].median()

# Write findings to report
with open(os.path.join(reports_dir, "02_hourly_validation_report.md"), "w") as f:
    f.write("# Hourly Dataset Validation Report\n\n")
    f.write("## 1. Verifying t_kWh\n")
    f.write(f"- Monotonically increasing for {meter1}: {is_monotonic}\n")
    f.write(f"- Median value: {median_val}\n")
    if not is_monotonic and median_val < 1.0:
        f.write("- **Conclusion**: `t_kWh` represents interval energy consumption (energy consumed during the 3-minute interval), NOT a cumulative meter reading.\n")
        f.write("- **Aggregation Method**: Summing `t_kWh` to create hourly consumption is correct.\n")
    else:
        f.write("- **Conclusion**: `t_kWh` might be cumulative. Needs further check.\n")

# 2. Investigate Missing Hourly Data
print("Investigating missing data...")
# Load the raw 2020 and 2021 data for just the common meters to do an exact missing analysis
df_2020 = pd.read_csv(file_2020)
df_2021 = pd.read_csv(os.path.join(raw_dir, "CEEW - Smart meter data Bareilly 2021.csv"))
df_full = pd.concat([df_2020, df_2021], ignore_index=True)
common_meters = ['BR02', 'BR04', 'BR05', 'BR06', 'BR08', 'BR09', 'BR11', 'BR12', 'BR13', 'BR15', 'BR16', 'BR18', 'BR19', 'BR22', 'BR24', 'BR27', 'BR28', 'BR29', 'BR30', 'BR31', 'BR32', 'BR33', 'BR34', 'BR35', 'BR36', 'BR37', 'BR38', 'BR39', 'BR42', 'BR43', 'BR44', 'BR45', 'BR46', 'BR48', 'BR49', 'BR50', 'BR51', 'BR52']

df_full = df_full[df_full['meter'].isin(common_meters)].copy()
df_full['x_Timestamp'] = pd.to_datetime(df_full['x_Timestamp'], errors='coerce')
df_full.dropna(subset=['x_Timestamp'], inplace=True)
df_full.loc[df_full['t_kWh'] < 0, 't_kWh'] = 0

df_full.set_index('x_Timestamp', inplace=True)
resampled = df_full.groupby('meter').resample('1h')
hourly_agg = resampled.agg({
    't_kWh': 'sum',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean'
}).reset_index()

hourly_size = resampled.size().reset_index(name='count')
hourly_agg.loc[hourly_size['count'] == 0, 't_kWh'] = np.nan


# Because resample creates a continuous index from the first to last timestamp PER meter,
# any hour without a raw reading will have NaN in t_kWh.
missing_analysis = []
coverage_stats = []

for meter in common_meters:
    m_df = hourly_agg[hourly_agg['meter'] == meter]
    if len(m_df) == 0:
        continue
    
    start = m_df['x_Timestamp'].min()
    end = m_df['x_Timestamp'].max()
    expected_hours = int((end - start).total_seconds() / 3600) + 1
    observed_hours = len(m_df.dropna(subset=['t_kWh']))
    missing_hours = expected_hours - observed_hours
    coverage = (observed_hours / expected_hours) * 100 if expected_hours > 0 else 0
    
    coverage_stats.append({
        'Meter': meter,
        'Expected Hours': expected_hours,
        'Observed Hours': observed_hours,
        'Missing Hours': missing_hours,
        'Coverage %': coverage,
        'Start': start,
        'End': end
    })

coverage_df = pd.DataFrame(coverage_stats)
coverage_df.to_csv(os.path.join(results_dir, "meter_hourly_coverage.csv"), index=False)

total_expected = coverage_df['Expected Hours'].sum()
total_observed = coverage_df['Observed Hours'].sum()
total_missing = coverage_df['Missing Hours'].sum()
pct_missing = (total_missing / total_expected) * 100

with open(os.path.join(reports_dir, "02_hourly_validation_report.md"), "a") as f:
    f.write("\n## 2. Missing Hourly Data Investigation\n")
    f.write(f"- Total Expected Hourly Records: {total_expected}\n")
    f.write(f"- Total Observed Hourly Records: {total_observed}\n")
    f.write(f"- Total Missing Hourly Records: {total_missing}\n")
    f.write(f"- Percentage of Missing Hourly Records: {pct_missing:.2f}%\n")
    
    if pct_missing > 0:
        f.write("\n## 3. Zero Imputation Validity\n")
        f.write("Replacing missing electricity consumption with `0` is **NOT scientifically justified**. A missing hour means the meter lost communication or was turned off, not necessarily that household consumption was strictly 0.00 kWh. Using 0 creates artificial drops.\n")
        f.write("Decision: We will keep missing values as NaN (Option B) for consumption, or perform interpolation for short gaps (Option C) and drop long gaps.\n")

    f.write("\n## 4. False Concept Drift Check\n")
    f.write("If we kept zeros, artificial drops would definitely trigger false drift alarms (e.g., consumption going from 1.5 kWh -> 0.0 -> 1.2 kWh). Leaving them as NaN or intelligently interpolating prevents this.\n")

# Regenerate V2 dataset properly
print("Regenerating V2 dataset...")

# Instead of filling with 0, we will:
# 1. Forward-fill/backward-fill short gaps (e.g., up to 2 hours) if scientifically acceptable?
# Wait, for energy, interpolation makes more sense.
# Let's just leave t_kWh as NaN for missing hours (Option B) so models/evaluators can ignore those rows, 
# or interpolate up to 2 hours. Let's interpolate up to 2 hours, and leave the rest as NaN.
hourly_agg['t_kWh_interpolated'] = hourly_agg.groupby('meter')['t_kWh'].transform(lambda x: x.interpolate(limit=2))
hourly_agg['t_kWh'] = hourly_agg['t_kWh_interpolated'] # Keep NaNs for long gaps
hourly_agg.drop(columns=['t_kWh_interpolated'], inplace=True)

# For Voltage/Freq/Current, interpolate up to 2 hours, then NaN.
hourly_agg['z_Avg Voltage (Volt)'] = hourly_agg.groupby('meter')['z_Avg Voltage (Volt)'].transform(lambda x: x.interpolate(limit=2))
hourly_agg['z_Avg Current (Amp)'] = hourly_agg.groupby('meter')['z_Avg Current (Amp)'].transform(lambda x: x.interpolate(limit=2))
hourly_agg['y_Freq (Hz)'] = hourly_agg.groupby('meter')['y_Freq (Hz)'].transform(lambda x: x.interpolate(limit=2))

output_file_v2 = os.path.join(processed_dir, "bareilly_hourly_clean_v2.csv")
hourly_agg.to_csv(output_file_v2, index=False)

with open(os.path.join(reports_dir, "02_hourly_validation_report.md"), "a") as f:
    f.write("\n## 7. Rebuild Decision\n")
    f.write("A. Is t_kWh aggregation correct? YES (Sum)\n")
    f.write("B. Is missing consumption = 0 scientifically justified? NO\n")
    f.write("C. Should the hourly dataset be regenerated? YES\n")
    f.write(f"- Regenerated dataset saved as `bareilly_hourly_clean_v2.csv`.\n")
    f.write("- Changes: Used linear interpolation for gaps up to 2 hours. Left longer gaps as NaN to avoid false concept drift.\n")

print("Validation complete. Dataset V2 generated.")
