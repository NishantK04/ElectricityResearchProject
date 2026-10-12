# Exploratory Data Analysis Report

## 1. Overall Consumption Statistics
- Mean: 0.3530 kWh
- Median: 0.2130 kWh
- Std Dev: 0.4514 kWh
- Min / Max: 0.0000 / 5.5260 kWh
- 25th / 75th Percentile: 0.0750 / 0.4380 kWh

## 2. Temporal Patterns
- **Hourly**: Peak consumption occurs around evening hours (18:00 - 22:00) and morning (07:00 - 09:00).
- **Weekly**: Slightly higher consumption on weekends compared to weekdays.
- **Monthly**: Clear seasonal pattern. Higher consumption during summer months (May-August) likely due to cooling load.

## 3. Household Differences
- Significant variation exists across the 38 households. The highest consuming household uses >3x the energy of the lowest consuming household, meaning models might need household-specific tuning or features.

## 4. Candidate Consumption-Pattern Change Periods
Based on the long-term trend analysis, we identify the following candidate concept-drift periods:

**Candidate Period 1: March 24, 2020 - May 31, 2020 (First COVID Lockdown)**
- **Observation**: Sudden increase and sustained high level of daytime consumption.
- **Why**: Entire families staying home during the day.
- **Nature**: Persistent for ~2 months, followed by gradual return to normal.

**Candidate Period 2: April 1, 2021 - June 15, 2021 (Second COVID Wave & Summer)**
- **Observation**: Sharp spike in consumption.
- **Why**: Combination of severe second wave lockdowns in UP (Bareilly) and peak summer cooling load.
- **Nature**: Highly volatile and persistent over the 2.5 months.

**Candidate Period 3: October/November Seasonal Transitions**
- **Observation**: Sharp drop in consumption.
- **Why**: Transition from summer/monsoon to winter, eliminating cooling loads.
- **Nature**: Gradual/Seasonal drift.

These periods will be crucial for evaluating the adaptation strategies in Phase 6-9.
