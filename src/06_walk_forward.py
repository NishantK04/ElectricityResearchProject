import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from river import drift

processed_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\data\processed"
results_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\results"

print("Loading dataset for Walk-Forward...")
df = pd.read_csv(os.path.join(processed_dir, "bareilly_ml_features.csv"))
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df = df.dropna().sort_values('x_Timestamp').reset_index(drop=True)

# Initial training period: Jan 1, 2020 to March 23, 2020
init_train_end = pd.to_datetime('2020-03-23 23:59:59')
train_df = df[df['x_Timestamp'] <= init_train_end]
sim_df = df[df['x_Timestamp'] > init_train_end]

features = [c for c in df.columns if c not in ['meter', 'x_Timestamp', 'target_next_hour']]
target = 'target_next_hour'

# We'll store all predictions here to analyze later
all_preds = []
update_events = []
adwin_events = []

def get_base_model():
    return RandomForestRegressor(n_estimators=5, max_depth=5, random_state=42, n_jobs=1)

# ---------------------------------------------------------
# 1. STATIC MODEL
# ---------------------------------------------------------
print("Running Strategy: Static...")
start_t = time.time()
model_static = get_base_model()
model_static.fit(train_df[features], train_df[target])
static_preds = model_static.predict(sim_df[features])

static_df = sim_df[['meter', 'x_Timestamp', target]].copy()
static_df.rename(columns={'x_Timestamp': 'timestamp', target: 'actual'}, inplace=True)
static_df['prediction'] = static_preds
static_df['method'] = 'Static'
static_df['configuration'] = 'None'
all_preds.append(static_df)
print(f"Static finished in {time.time() - start_t:.1f}s")

# ---------------------------------------------------------
# Helper for chunked prediction (Periodic and Sliding)
# ---------------------------------------------------------
def run_chunked_strategy(method_name, config, interval_days, window_days=None):
    print(f"Running Strategy: {method_name} ({config})...")
    start_t = time.time()
    
    current_time = init_train_end
    max_time = sim_df['x_Timestamp'].max()
    
    # Initial model
    model = get_base_model()
    if window_days:
        window_start = current_time - pd.Timedelta(days=window_days)
        curr_train = df[(df['x_Timestamp'] > window_start) & (df['x_Timestamp'] <= current_time)]
    else:
        curr_train = df[df['x_Timestamp'] <= current_time]
        
    model.fit(curr_train[features], curr_train[target])
    
    updates = 0
    total_train_time = 0
    
    while current_time < max_time:
        next_time = current_time + pd.Timedelta(days=interval_days)
        chunk = df[(df['x_Timestamp'] > current_time) & (df['x_Timestamp'] <= next_time)]
        
        if len(chunk) > 0:
            preds = model.predict(chunk[features])
            c_df = chunk[['meter', 'x_Timestamp', target]].copy()
            c_df.rename(columns={'x_Timestamp': 'timestamp', target: 'actual'}, inplace=True)
            c_df['prediction'] = preds
            c_df['method'] = method_name
            c_df['configuration'] = config
            all_preds.append(c_df)
        
        current_time = next_time
        
        # Retrain
        if current_time < max_time:
            t0 = time.time()
            if window_days:
                window_start = current_time - pd.Timedelta(days=window_days)
                curr_train = df[(df['x_Timestamp'] > window_start) & (df['x_Timestamp'] <= current_time)]
            else:
                curr_train = df[df['x_Timestamp'] <= current_time]
            
            if len(curr_train) > 0:
                model = get_base_model()
                model.fit(curr_train[features], curr_train[target])
                t_train = time.time() - t0
                updates += 1
                total_train_time += t_train
                
                update_events.append({
                    'method': method_name,
                    'configuration': config,
                    'update_date': current_time,
                    'training_samples': len(curr_train),
                    'training_time': t_train
                })
    
    print(f"Finished in {time.time() - start_t:.1f}s. Updates: {updates}")

# ---------------------------------------------------------
# 2. PERIODIC RETRAINING
# ---------------------------------------------------------
run_chunked_strategy('Periodic', '7-day', interval_days=7, window_days=None)
run_chunked_strategy('Periodic', '14-day', interval_days=14, window_days=None)
run_chunked_strategy('Periodic', '30-day', interval_days=30, window_days=None)

# ---------------------------------------------------------
# 3. SLIDING WINDOW
# ---------------------------------------------------------
run_chunked_strategy('Sliding', '7-day', interval_days=7, window_days=7)
run_chunked_strategy('Sliding', '14-day', interval_days=14, window_days=14)
run_chunked_strategy('Sliding', '30-day', interval_days=30, window_days=30)

# ---------------------------------------------------------
# 4. ADWIN DRIFT-BASED UPDATING
# ---------------------------------------------------------
print("Running Strategy: ADWIN Drift-Based...")
start_t = time.time()

# We will step day-by-day for speed, but evaluate ADWIN hour-by-hour within the day.
current_time = init_train_end
max_time = sim_df['x_Timestamp'].max()

model_adwin = get_base_model()
curr_train = df[df['x_Timestamp'] <= current_time]
model_adwin.fit(curr_train[features], curr_train[target])

# ADWIN detector monitoring Absolute Error
# We'll use a single global ADWIN that monitors the daily mean absolute error across all meters
adwin_detector = drift.ADWIN(delta=0.002) 

updates = 0
total_train_time = 0
last_retrain_time = current_time

while current_time < max_time:
    next_time = current_time + pd.Timedelta(days=1)
    chunk = df[(df['x_Timestamp'] > current_time) & (df['x_Timestamp'] <= next_time)]
    
    if len(chunk) > 0:
        # Predict for this day in one shot
        preds = model_adwin.predict(chunk[features])
        
        # Store preds
        c_df = chunk[['meter', 'x_Timestamp', target]].copy()
        c_df.rename(columns={'x_Timestamp': 'timestamp', target: 'actual'}, inplace=True)
        c_df['prediction'] = preds
        c_df['method'] = 'ADWIN'
        c_df['configuration'] = 'Global MAE'
        all_preds.append(c_df)
        
        # Calculate daily mean absolute error
        daily_mae = np.mean(np.abs(chunk[target].values - preds))
        
        # Update ADWIN once per day
        adwin_detector.update(daily_mae)
        
        if adwin_detector.drift_detected:
            # Drift detected! Retrain if cooldown passed (e.g. 7 days to prevent freezing)
            if (next_time - last_retrain_time).days >= 7:
                t0 = time.time()
                curr_train = df[df['x_Timestamp'] <= next_time]
                model_adwin = get_base_model()
                model_adwin.fit(curr_train[features], curr_train[target])
                t_train = time.time() - t0
                
                updates += 1
                total_train_time += t_train
                last_retrain_time = next_time
                
                adwin_events.append({
                    'detection_timestamp': next_time,
                    'error_before_detection': daily_mae,
                    'update_triggered': True,
                    'training_samples': len(curr_train),
                    'retraining_time': t_train
                })
                
                update_events.append({
                    'method': 'ADWIN',
                    'configuration': 'Global MAE',
                    'update_date': next_time,
                    'training_samples': len(curr_train),
                    'training_time': t_train
                })
            else:
                adwin_events.append({
                    'detection_timestamp': next_time,
                    'error_before_detection': daily_mae,
                    'update_triggered': False,
                    'training_samples': 0,
                    'retraining_time': 0
                })
    
    current_time = next_time

print(f"ADWIN finished in {time.time() - start_t:.1f}s. Updates: {updates}")

# ---------------------------------------------------------
# Save Raw Results
# ---------------------------------------------------------
print("Saving walk-forward predictions...")
preds_df = pd.concat(all_preds, ignore_index=True)
preds_df['error'] = preds_df['actual'] - preds_df['prediction']
preds_df['absolute_error'] = preds_df['error'].abs()

# Save as parquet to save space and I/O time
preds_df.to_parquet(os.path.join(results_dir, "walk_forward_predictions.parquet"), index=False)

pd.DataFrame(update_events).to_csv(os.path.join(results_dir, "update_events.csv"), index=False)
pd.DataFrame(adwin_events).to_csv(os.path.join(results_dir, "adwin_events.csv"), index=False)

print("Phase 6 Walk-Forward Generation Complete.")
