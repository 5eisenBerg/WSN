import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# --- Configuration ---
# Make sure these file names match the output of your experiment
TRAINING_LOG_FILE = 'training_log_v53.csv'
FINAL_RESULTS_FILE = 'final_results_v53.xlsx'

# Set a professional plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# --- 1. Plot Training Convergence Curve ---
try:
    log_df = pd.read_csv(TRAINING_LOG_FILE)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=log_df, x='epoch', y='score', marker='o', label='Validation Score per Epoch')
    
    # Highlight the best score
    best_epoch = log_df['score'].idxmax()
    best_score = log_df['score'].max()
    plt.axvline(x=log_df['epoch'][best_epoch], color='r', linestyle='--', label=f'Best Model @ Epoch {log_df["epoch"][best_epoch]} (Score: {best_score:.4f})')
    
    plt.title('DQN Agent Learning Convergence', fontsize=16)
    plt.xlabel('Training Epoch')
    plt.ylabel('Validation Score (PDR & HP Delay Composite)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_convergence_v53.png', dpi=300)
    print("✅ Successfully generated 'training_convergence_v53.png'")
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: Could not find '{TRAINING_LOG_FILE}'. Please make sure it's in the correct directory.")
except Exception as e:
    print(f"An error occurred while plotting the training curve: {e}")


# --- 2. Plot Final Performance Bar Chart ---
try:
    # The xlsx file has the policy names as the index
    results_df = pd.read_excel(FINAL_RESULTS_FILE, index_col=0)
    
    # We need to parse the 'mean±ci' strings into separate columns for plotting
    metrics_to_plot = {
        'throughput_kbps': 'Throughput (kbps)',
        'pdr_pct': 'PDR (%)',
        'avg_normal_delay_s': 'Avg. Normal Delay (s)',
        'avg_hp_delay_s': 'Avg. HP Delay (s)',
        'energy_nj_bit': 'Energy (nJ/bit)'
    }
    
    plot_data = []
    for policy in results_df.index:
        for metric_col, display_name in metrics_to_plot.items():
            if metric_col in results_df.columns:
                mean_ci = results_df.loc[policy, metric_col].split('±')
                mean = float(mean_ci[0])
                ci = float(mean_ci[1])
                plot_data.append([policy, display_name, mean, ci])

    plot_df = pd.DataFrame(plot_data, columns=['Policy', 'Metric', 'Mean', 'CI'])

    # Create faceted bar charts for each metric
    g = sns.catplot(
        data=plot_df, kind="bar",
        x="Policy", y="Mean", col="Metric",
        ci=None, height=5, aspect=0.9, col_wrap=3,
        sharey=False # Each metric has its own y-axis scale
    )
    
    # Add error bars manually
    for ax in g.axes.flat:
        for i, bar in enumerate(ax.patches):
            # Find the corresponding data point
            policy = ax.get_xticklabels()[i].get_text()
            metric = ax.get_title().split(' = ')[1]
            
            point = plot_df[(plot_df['Policy'] == policy) & (plot_df['Metric'] == metric)]
            if not point.empty:
                mean = point['Mean'].values[0]
                ci = point['CI'].values[0]
                ax.errorbar(x=i, y=mean, yerr=ci, fmt='none', c='black', capsize=5)

    g.fig.suptitle('Performance Comparison of Scheduling Policies', y=1.03, fontsize=18)
    g.set_axis_labels("", "Mean Value")
    g.set_titles("{col_name}")
    g.tight_layout()
    plt.savefig('performance_comparison_v53.png', dpi=300)
    print("✅ Successfully generated 'performance_comparison_v53.png'")
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: Could not find '{FINAL_RESULTS_FILE}'. Please make sure it's in the correct directory.")
except Exception as e:
    print(f"An error occurred while plotting the performance comparison: {e}")

# --- 3. Plot Delay Distribution (CDF) ---
# This requires re-running the simulation to collect per-packet delays, 
# as this data is not in the final summary. We will simulate one run for each policy.
# NOTE: This part of the script assumes the main V53 training script is available
# and can be imported. For simplicity, we will just show the structure.

print("\n--- Delay Distribution Analysis ---")
print("NOTE: Generating CDF plots requires re-running simulations to get per-packet data.")
print("This can be added as a separate, more detailed analysis script.")
# A full implementation would involve:
# 1. Modifying the C++ to dump every packet's delay to a file.
# 2. Running one evaluation episode for each policy.
# 3. Loading the delay data files here.
# 4. Using seaborn's `ecdfplot` to plot the CDF.
# Example:
# delays_dqn = pd.read_csv('dqn_delays.csv')
# delays_sp = pd.read_csv('sp_delays.csv')
# sns.ecdfplot(data=delays_dqn, x='delay', label='DQN-Edge')
# sns.ecdfplot(data=delays_sp, x='delay', label='Strict Priority')
# plt.xlabel('Packet End-to-End Delay (s)')
# plt.ylabel('Probability (CDF)')
# plt.title('CDF of Packet Delays')
# plt.legend()
# plt.show()
