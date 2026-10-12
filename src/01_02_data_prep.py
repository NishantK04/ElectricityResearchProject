import pandas as pd
import numpy as np
import os

# Set paths
raw_dir = r"C:\Users\nisha\Downloads\ML-Based Household Electricity consumption prediction\ElectricityProject"
processed_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\data\processed"
reports_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\reports"

file_2020 = os.path.join(raw_dir, "CEEW - Smart meter data Bareilly 2020.csv")
file_2021 = os.path.join(raw_dir, "CEEW - Smart meter data Bareilly 2021.csv")

# ---------------------------------------------------------
# STEP 1: DATA INSPECTION
# ---------------------------------------------------------
print("Starting Step 1: Data Inspection...")
df_2020 = pd.read_csv(file_2020)
df_2021 = pd.read_csv(file_2021)

def inspect_df(df, name):
    # Basic info
    rows, cols = df.shape
    columns = list(df.columns)
    dtypes = df.dtypes.to_dict()
    
    # Meter info
    meters = set(df['meter'].unique())
    
    # Time info
    df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'], errors='coerce')
    min_date = df['x_Timestamp'].min()
    max_date = df['x_Timestamp'].max()
    
    # Missing and duplicates
    missing = df.isnull().sum().to_dict()
    dups = df.duplicated(subset=['meter', 'x_Timestamp']).sum()
    
    # t_kWh stats
    t_min = df['t_kWh'].min()
    t_max = df['t_kWh'].max()
    t_mean = df['t_kWh'].mean()
    t_median = df['t_kWh'].median()
    
    # invalid (negative consumption)
    invalid_pct = (df['t_kWh'] < 0).sum() / len(df) * 100
    
    report = {
        "File": name,
        "Rows": rows,
        "Columns": cols,
        "Column Names": columns,
        "Data Types": {k: str(v) for k, v in dtypes.items()},
        "Unique Meters": len(meters),
        "Date Range": f"{min_date} to {max_date}",
        "Sampling Interval (approx)": "3 minutes",
        "Missing Values": missing,
        "Duplicate Timestamps": int(dups),
        "t_kWh (min/max/mean/median)": f"{t_min} / {t_max} / {t_mean:.4f} / {t_median:.4f}",
        "Invalid values (%)": f"{invalid_pct:.4f}%",
        "meters_set": meters
    }
    return report

report_2020 = inspect_df(df_2020, "Bareilly 2020")
report_2021 = inspect_df(df_2021, "Bareilly 2021")

common_meters = report_2020.pop('meters_set').intersection(report_2021.pop('meters_set'))
report_text = f"""# Data Inspection Report

## {report_2020['File']}
- Rows: {report_2020['Rows']}
- Columns: {report_2020['Columns']}
- Column Names: {report_2020['Column Names']}
- Data Types: {report_2020['Data Types']}
- Unique Meters: {report_2020['Unique Meters']}
- Date Range: {report_2020['Date Range']}
- Missing Values: {report_2020['Missing Values']}
- Duplicate Timestamps: {report_2020['Duplicate Timestamps']}
- Electricity (t_kWh) min/max/mean/median: {report_2020['t_kWh (min/max/mean/median)']}
- Invalid values (<0): {report_2020['Invalid values (%)']}

## {report_2021['File']}
- Rows: {report_2021['Rows']}
- Columns: {report_2021['Columns']}
- Column Names: {report_2021['Column Names']}
- Data Types: {report_2021['Data Types']}
- Unique Meters: {report_2021['Unique Meters']}
- Date Range: {report_2021['Date Range']}
- Missing Values: {report_2021['Missing Values']}
- Duplicate Timestamps: {report_2021['Duplicate Timestamps']}
- Electricity (t_kWh) min/max/mean/median: {report_2021['t_kWh (min/max/mean/median)']}
- Invalid values (<0): {report_2021['Invalid values (%)']}

## Overlap
- Households present in both years: {len(common_meters)}
- Common Meters: {sorted(list(common_meters))}
"""

with open(os.path.join(reports_dir, "01_data_inspection_report.md"), "w") as f:
    f.write(report_text)
print("Inspection report saved.")


# ---------------------------------------------------------
# STEP 2: DATA CLEANING
# ---------------------------------------------------------
print("Starting Step 2: Data Cleaning...")
# Combine datasets
df = pd.concat([df_2020, df_2021], ignore_index=True)

# Keep only common meters
df = df[df['meter'].isin(common_meters)].copy()

# Sort by meter and timestamp
df.sort_values(by=['meter', 'x_Timestamp'], inplace=True)

# Drop duplicates
df.drop_duplicates(subset=['meter', 'x_Timestamp'], keep='first', inplace=True)

# Handle impossible negative electricity values
df.loc[df['t_kWh'] < 0, 't_kWh'] = 0

print(f"Cleaned dataset rows: {len(df)}")


# ---------------------------------------------------------
# STEP 3: HOURLY AGGREGATION
# ---------------------------------------------------------
print("Starting Step 3: Hourly Aggregation...")
# t_kWh is interval consumption (e.g., 0.001 kWh in 3 mins). 
# We need to SUM t_kWh for hourly total.
# For Voltage, Current, Freq, we take the MEAN.

df.set_index('x_Timestamp', inplace=True)

# Group by meter, then resample to 1H
hourly_df = df.groupby('meter').resample('1H').agg({
    't_kWh': 'sum',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean'
}).reset_index()

# Because resampling might introduce NaNs for hours with no data, let's fill them
# We can forward-fill up to 2 hours, then fill remaining with 0 or drop.
# For simplicity and given 0 missing values initially, there shouldn't be large gaps.
# Let's check missing after resample
print("Missing values after hourly resampling:")
print(hourly_df.isnull().sum())

# Fill NaNs
hourly_df['t_kWh'] = hourly_df['t_kWh'].fillna(0)
hourly_df['z_Avg Voltage (Volt)'] = hourly_df['z_Avg Voltage (Volt)'].ffill().bfill()
hourly_df['z_Avg Current (Amp)'] = hourly_df['z_Avg Current (Amp)'].fillna(0)
hourly_df['y_Freq (Hz)'] = hourly_df['y_Freq (Hz)'].ffill().bfill()

# Save processed hourly data
output_file = os.path.join(processed_dir, "bareilly_hourly_clean.csv")
hourly_df.to_csv(output_file, index=False)
print(f"Hourly dataset saved to {output_file}. Rows: {len(hourly_df)}")
print("Phases 1 and 2 complete.")
