import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from math import pi

# --- V63 Config ---
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
DATA_DIR = 'data_v62'
FIG_DIR = 'figures_v63'
TRAIN_LOG = 'training_log_v62.csv'
RAW_RESULTS = 'raw_results_v62.csv'
POLICIES = ['Proposed DQN-Edge', 'Strict Priority', 'Random']
POLICY_MAP = {"Proposed DQN-Edge": "DQN", "Strict Priority": "StrictPriority", "Random": "Random"}
ACTION_LABELS = ['Send HP', 'Send Normal', 'Drop Normal', 'Sleep', 'Drop HP']
METRICS_TO_PLOT = {'throughput_kbps': 'Throughput (kbps)', 'pdr_pct': 'PDR (%)', 'avg_hp_delay_s': 'Avg. HP Delay (s)', 'energy_nj_bit': 'Energy (nJ/bit)'}

# --- Plotting Functions ---
def plot_learning_curves(log_file):
    try:
        df = pd.read_csv(log_file)
        df['score_smooth'] = df['score'].rolling(window=5, min_periods=1, center=True).mean()
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        ax1.plot(df['epoch'], df['score_smooth'], 'b-', linewidth=3, label='Validation Score (5-Epoch Avg)')
        ax2.plot(df['epoch'], df['epsilon'], 'r--', alpha=0.6, label='Epsilon Decay')
        ax1.set_xlabel('Training Epoch'); ax1.set_ylabel('Composite Validation Score', color='b'); ax2.set_ylabel('Epsilon', color='r')
        best_epoch_idx = df['score'].idxmax(); best_score = df['score'].max(); best_epoch = df['epoch'][best_epoch_idx]
        plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model @ Epoch {best_epoch} (Score: {best_score:.3f})')
        plt.scatter(best_epoch, best_score, color='green', s=120, zorder=5, label='Best Score Point')
        plt.title('Agent Training Progression', fontsize=18, fontweight='bold'); fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); fig.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'pub_fig_1_learning_curve.png'), dpi=300, bbox_inches="tight"); plt.close()
        print("✅ Figure 1: Smoothed Learning Curve generated.")
    except Exception as e: print(f"❌ Could not generate Learning Curve: {e}")

def plot_reward_curve():
    try:
        reward_files = [f for f in os.listdir(DATA_DIR) if 'reward_log' in f]
        all_rewards = [np.loadtxt(os.path.join(DATA_DIR, f)) for f in reward_files]
        if not all_rewards: raise FileNotFoundError("No reward logs found.")
        mean_rewards = pd.DataFrame(all_rewards).mean(axis=0)
        smoothed_rewards = mean_rewards.rolling(window=100, min_periods=1).mean()
        plt.figure(figsize=(12, 7))
        plt.plot(smoothed_rewards)
        plt.title('Average Per-Step Reward Across All Training Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step'); plt.ylabel('Smoothed Reward (100-step avg)')
        plt.savefig(os.path.join(FIG_DIR, 'pub_fig_2_reward_curve.png'), dpi=300, bbox_inches="tight"); plt.close()
        print("✅ Figure 2: Reward Curve generated.")
    except Exception as e: print(f"❌ Could not generate Reward Curve: {e}")

def plot_performance_boxplots(raw_df):
    try:
        df_melted = raw_df.melt(id_vars='Policy', value_vars=METRICS_TO_PLOT.keys(), var_name='Metric', value_name='Value')
        df_melted['Metric'] = df_melted['Metric'].map(METRICS_TO_PLOT)
        g = sns.catplot(data=df_melted, kind='box', x='Policy', y='Value', col='Metric', col_wrap=2, height=5, aspect=1.3, sharey=False, order=POLICIES)
        g.fig.suptitle('Distribution of Performance Metrics Across 20 Runs', y=1.03, fontsize=18, fontweight='bold')
        g.set_axis_labels("", "Metric Value"); g.set_titles("{col_name}")
        plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(FIG_DIR, 'pub_fig_3_boxplots.png'), dpi=300, bbox_inches="tight"); plt.close()
        print("✅ Figure 3: Performance Boxplots generated.")
    except Exception as e: print(f"❌ Could not generate Boxplots: {e}")

def plot_delay_cdf():
    try:
        plt.figure(figsize=(12, 7))
        for policy_label, policy_short in POLICY_MAP.items():
            hp_delays = np.loadtxt(os.path.join(DATA_DIR, f'{policy_short}_hp_delays.txt'))
            normal_delays = np.loadtxt(os.path.join(DATA_DIR, f'{policy_short}_normal_delays.txt'))
            sns.ecdfplot(hp_delays, label=f'{policy_label} (HP)', linewidth=3)
            sns.ecdfplot(normal_delays, label=f'{policy_label} (Normal)', linestyle='--', linewidth=3)
        plt.title('CDF of Packet End-to-End Delays', fontsize=18, fontweight='bold')
        plt.xlabel('Delay (seconds)'); plt.ylabel('Probability (CDF)'); plt.legend(); plt.grid(True, linestyle=':'); plt.xlim(left=0)
        plt.savefig(os.path.join(FIG_DIR, 'pub_fig_4_delay_cdf.png'), dpi=300, bbox_inches="tight"); plt.close()
        print("✅ Figure 4: Delay CDF generated.")
    except Exception as e: print(f"❌ Could not generate Delay CDF: {e}")

def plot_radar_chart(df):
    try:
        metrics = ['pdr_pct', 'throughput_kbps', 'avg_hp_delay_s', 'energy_nj_bit']
        labels = ['PDR (%) ↑', 'Throughput (kbps) ↑', 'HP Delay (s) ↓', 'Energy (nJ/bit) ↓']
        
        df_norm = df.copy()
        for metric in metrics:
            min_val, max_val = df_norm[metric].min(), df_norm[metric].max()
            if metric in ['avg_hp_delay_s', 'energy_nj_bit']: # Lower is better, so invert
                df_norm[metric] = (max_val - df_norm[metric]) / (max_val - min_val) if (max_val - min_val) != 0 else 0
            else: # Higher is better
                df_norm[metric] = (df_norm[metric] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0

        angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]; angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True)); ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], labels); ax.set_rlabel_position(0); plt.yticks([0.25,0.5,0.75], ["0.25","0.50","0.75"], color="grey", size=7); plt.ylim(0,1)

        for i, row in df_norm.iterrows():
            data = row[metrics].values.flatten().tolist(); data += data[:1]
            ax.plot(angles, data, linewidth=2, linestyle='solid', label=row['Policy'])
            ax.fill(angles, data, alpha=0.25)
        
        plt.title('Multi-Objective Performance Radar Chart', size=20, color='black', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)); plt.savefig(os.path.join(FIG_DIR,'pub_fig_5_radar_chart.png'),dpi=300,bbox_inches="tight"); plt.close()
        print("✅ Figure 5: Radar Chart generated.")
    except Exception as e: print(f"❌ Could not generate Radar Chart: {e}")

def generate_stats_table(raw_df):
    print("\n\n--- 📋 Statistical Significance Table ---")
    try:
        dqn_data = raw_df[raw_df['Policy'] == 'Proposed DQN-Edge']
        sp_data = raw_df[raw_df['Policy'] == 'Strict Priority']
        
        stats_results = []
        for metric, name in METRICS_TO_PLOT.items():
            # Shapiro-Wilk test for normality
            _, p_norm_dqn = stats.shapiro(dqn_data[metric])
            _, p_norm_sp = stats.shapiro(sp_data[metric])
            
            if p_norm_dqn > 0.05 and p_norm_sp > 0.05:
                # Welch's t-test if both are normal
                t_stat, p_val = stats.ttest_ind(dqn_data[metric], sp_data[metric], equal_var=False)
                test_used = "Welch's t-test"
            else:
                # Mann-Whitney U test if not normal
                u_stat, p_val = stats.mannwhitneyu(dqn_data[metric], sp_data[metric], alternative='two-sided')
                test_used = "Mann-Whitney U"

            # Cohen's d for effect size
            n1, n2 = len(dqn_data), len(sp_data)
            s1, s2 = dqn_data[metric].std(ddof=1), sp_data[metric].std(ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            cohen_d = (dqn_data[metric].mean() - sp_data[metric].mean()) / pooled_std if pooled_std > 0 else 0
            
            stats_results.append({'Metric': name, 'p-value (vs. SP)': f"{p_val:.3e}", "Cohen's d": f"{cohen_d:.4f}", "Test Used": test_used})
        
        stats_df = pd.DataFrame(stats_results).set_index('Metric')
        print(stats_df.to_markdown())
        print("\n" + stats_df.to_latex())
    except Exception as e: print(f"❌ Could not generate Stats Table: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR) or not os.path.exists(TRAIN_LOG) or not os.path.exists(RAW_RESULTS):
        print("--- ⚠️ Data files not found. Please run the main experiment script first. ---")
    else:
        os.makedirs(FIG_DIR, exist_ok=True)
        raw_df = pd.read_csv(RAW_RESULTS)
        
        plot_learning_curves(TRAIN_LOG)
        plot_reward_curve()
        plot_performance_boxplots(raw_df)
        plot_delay_cdf()
        
        summary_df = raw_df.groupby('Policy').mean().reset_index()
        plot_radar_chart(summary_df)
        
        generate_stats_table(raw_df)

        print(f"\n\n✅ All figures and tables have been generated in the '{FIG_DIR}' directory.")
