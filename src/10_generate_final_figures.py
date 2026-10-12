import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

results_dir = r"C:\Users\nisha\.gemini\antigravity\scratch\ElectricityResearchProject\results"
figures_dir = os.path.join(results_dir, "figures", "final")

metrics_df = pd.read_csv(os.path.join(results_dir, "phase10_recalculated_metrics.csv"))
timing_df = pd.read_csv(os.path.join(results_dir, "phase9_timing_comparison.csv"))

merged_df = pd.merge(metrics_df, timing_df, on='Strategy')

# Strategy clean naming
merged_df['Strategy_Clean'] = merged_df['Strategy'].str.replace('ADWIN (delta=0.001)', 'ADWIN', regex=False)

plt.style.use('seaborn-v0_8-whitegrid')

# 1. MAE Comparison
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=merged_df, x='Strategy_Clean', y='MAE', palette='viridis')
plt.title('Prediction Accuracy (Mean Absolute Error) by Strategy', fontsize=14)
plt.ylabel('MAE (kWh)', fontsize=12)
plt.xlabel('Updating Strategy', fontsize=12)
for i in ax.containers:
    ax.bar_label(i, fmt='%.4f', padding=3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "final_mae_comparison.png"), dpi=300)
plt.close()

# 2. R2 Comparison
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=merged_df, x='Strategy_Clean', y='R2', palette='magma')
plt.title('Explained Variance (R²) by Strategy', fontsize=14)
plt.ylabel('R² Score', fontsize=12)
plt.xlabel('Updating Strategy', fontsize=12)
for i in ax.containers:
    ax.bar_label(i, fmt='%.4f', padding=3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "final_r2_comparison.png"), dpi=300)
plt.close()

# 3. Accuracy vs Cost
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='Total Retrain Time (s)', y='MAE', hue='Strategy_Clean', s=300, palette='Set1')
for i in range(len(merged_df)):
    plt.text(merged_df['Total Retrain Time (s)'].iloc[i] + 0.2, merged_df['MAE'].iloc[i], 
             merged_df['Strategy_Clean'].iloc[i], fontsize=11)
plt.title('Accuracy (MAE) vs. Retraining Cost', fontsize=14)
plt.xlabel('Total Retraining Time (seconds)', fontsize=12)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "final_accuracy_vs_cost.png"), dpi=300)
plt.close()

print("Figures successfully generated in results/figures/final")
