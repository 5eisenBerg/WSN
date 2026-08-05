import sys
import os
import subprocess
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import shutil
from collections import deque
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from per import PrioritizedReplayBuffer

# --- V58b Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(CURRENT_DIR), 'networks'))
from network import DQN

NS3_PATH = '/home/heisenberg/ns3-workspace/bake/source/ns-3.40'
SIM_SCRIPT = 'wsn_100dynamic'
PORT = 5555
MODEL_PATH = os.path.join(CURRENT_DIR, "v58_best.pth")

# V58b: Focused scheduling state and action sizes
STATE_SIZE, ACTION_SIZE = 6, 5
TRAINING_EPOCHS, TRAIN_STEPS_PER_EPOCH, EVAL_STEPS, BASE_SEED = 30, 2500, 5000, 12345

# --- Reward Weights ---
HP_REWARD = 100
NORMAL_REWARD = 20
DROP_HP_PENALTY = -300
DROP_NORMAL_PENALTY = -50
SLEEP_PENALTY = -5
HP_DELAY_WEIGHT = -40
NORMAL_DELAY_WEIGHT = -20

random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PER_DoubleDQNAgent:
    def __init__(self, tau=0.005, warmup=1000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.model = DQN(STATE_SIZE, ACTION_SIZE)
        self.target = DQN(STATE_SIZE, ACTION_SIZE)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = PrioritizedReplayBuffer(100000, alpha)
        self.epsilon = 1.0
        self.train_steps = 0
        self.tau = tau
        self.warmup = warmup
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=TRAINING_EPOCHS * TRAIN_STEPS_PER_EPOCH)

    def act(self, s):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(ACTION_SIZE)
        self.model.eval()
        with torch.no_grad():
            action = self.model(torch.FloatTensor(s).unsqueeze(0)).argmax().item()
        self.model.train()
        return action

    def train(self):
        if len(self.memory) < self.warmup:
            return
        self.model.train()
        beta = min(1.0, self.beta_start + self.train_steps * (1.0 - self.beta_start) / self.beta_frames)
        samples, idx, w = self.memory.sample(128, beta)
        s, a, r, ns, d = zip(*samples)
        
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns))
        d = torch.FloatTensor(d).unsqueeze(1)
        w = torch.FloatTensor(w).unsqueeze(1)

        curr_q = self.model(s).gather(1, a)
        with torch.no_grad():
            next_a = self.model(ns).argmax(1).unsqueeze(1)
            target_q = r + 0.99 * self.target(ns).gather(1, next_a) * (1 - d)
            
        td_error = torch.abs(target_q - curr_q)
        loss = (w * F.smooth_l1_loss(curr_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        prios = td_error.detach().cpu().numpy().flatten() + 1e-5
        self.memory.update_priorities(idx, prios)
        
        if len(self.memory) > self.warmup:
            self.epsilon = max(0.05, self.epsilon * 0.9995)
        self._soft_update_target()
        self.train_steps += 1

    def _soft_update_target(self):
        for tp, lp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)
        self.target.eval()

    def remember(self, s, a, r, ns, d):
        self.memory.add(s, a, r, ns, d)

    def save(self, p):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "step": self.train_steps
        }, p)

    def load(self, p):
        c = torch.load(p, map_location="cpu")
        self.model.load_state_dict(c["model"])
        self.optimizer.load_state_dict(c["optimizer"])
        self.scheduler.load_state_dict(c["scheduler"])
        self.epsilon = c["epsilon"]
        self.train_steps = c["step"]
        self.target.load_state_dict(self.model.state_dict())


def calculate_reward(s, ps, info):
    r = info.get('hp_sent', 0) * HP_REWARD + info.get('normal_sent', 0) * NORMAL_REWARD
    r += info.get('hp_dropped', 0) * DROP_HP_PENALTY + info.get('normal_dropped', 0) * DROP_NORMAL_PENALTY
    r += info.get('hp_timeout', 0) * DROP_HP_PENALTY * 1.2 + info.get('normal_timeout', 0) * DROP_NORMAL_PENALTY * 1.2
    r += HP_DELAY_WEIGHT * s[4] + NORMAL_DELAY_WEIGHT * s[3]
    
    ec = ps[2] - s[2]
    if ec > 0:
        r -= ec * 5000
    if info.get('action') == 3:
        r += SLEEP_PENALTY
    return np.tanh(r / 250.0)


def run_simulation(policy, steps, train_mode=False, run_id=1, agent=None, seed=BASE_SEED, collect_data=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from ns3gym import ns3env
    
    cmd = f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT} --run={run_id}'"
    p = subprocess.Popen(cmd, cwd=NS3_PATH, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    
    env = None
    original_epsilon = None
    actions_log = []
    try:
        env = ns3env.Ns3Env(port=PORT, startSim=False)
        s = np.array(env.reset(), dtype=np.float32)
        if agent is None:
            agent = PER_DoubleDQNAgent()
            if policy == "DQN" and not train_mode:
                try:
                    agent.load(MODEL_PATH)
                    agent.epsilon = 0.0
                except FileNotFoundError:
                    print("⚠️ Model not found. DQN is RANDOM.")
                    policy = "Random"
        elif policy == "DQN" and not train_mode:
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
        
        final_info = {}
        bar = tqdm(range(steps), desc=f"{'Eval' if not train_mode else 'Train'} {policy} Run {run_id}")
        for step in bar:
            ps = s.copy()
            if train_mode and len(agent.memory) < agent.warmup:
                a = np.random.randint(ACTION_SIZE)
            elif policy == "DQN":
                a = agent.act(s)
            elif policy == "StrictPriority":
                if s[1] > 0:
                    a = 0
                elif s[0] > 0:
                    a = 1
                else:
                    a = 3
            else:
                a = np.random.randint(ACTION_SIZE)
            
            if collect_data:
                actions_log.append(a)
                
            s_next, _, d, info_str = env.step(a)
            info = json.loads(info_str if info_str else '{}')
            info['action'] = a
            s_next = np.array(s_next, dtype=np.float32)
            
            if train_mode:
                r = 0 if d else calculate_reward(s_next, ps, info)
                agent.remember(s, a, r, s_next, d)
                agent.train()
                
            s = s_next
            if d:
                assert "throughput_kbps" in info, "Final metrics not in terminal info block"
                final_info = info
                break
                
        if collect_data:
            datadir = os.path.join(CURRENT_DIR, 'data_v58')
            os.makedirs(datadir, exist_ok=True)
            for fname in ['hp_delays.txt', 'normal_delays.txt', 'queue_log.txt', 'energy_log.txt']:
                src_path = os.path.join(NS3_PATH, fname)
                if os.path.exists(src_path):
                    shutil.copy(src_path, os.path.join(datadir, f'{policy}_{fname}'))
            np.savetxt(os.path.join(datadir, f'{policy}_actions.txt'), actions_log, fmt='%d')
            
        return (final_info, agent)
    finally:
        if original_epsilon is not None and agent is not None:
            agent.epsilon = original_epsilon
        if env is not None:
            try: env.close()
            except: pass
        try: p.kill()
        except: pass
        subprocess.run(f"pkill -9 -f {SIM_SCRIPT}", shell=True, check=False)
        time.sleep(1)


def get_eval_score(res):
    pdr = res.get('pdr_pct', 0)
    plr = res.get('plr_pct', 100)
    hd = res.get('avg_hp_delay_s', 5)
    score = (0.5 * pdr / 100.0) + (0.2 * (1 - plr / 100.0)) + (0.3 * (1 - min(hd / 2.0, 1.0)))
    return np.clip(score, 0.0, 1.0)


def main_experiment():
    print("🛠️ V58b: Rebuilding NS-3...")
    build = subprocess.run(f"./ns3 build", cwd=NS3_PATH, shell=True, capture_output=True, text=True)
    if build.returncode != 0:
        print(f"❌ Build Fail:\n{build.stderr}")
        sys.exit(1)
    print("✅ Build OK.")
    
    training_agent = PER_DoubleDQNAgent()
    best_eval_score = float("-inf")
    best_epoch = 0
    train_log = []
    
    print("\n🧠 V58b: Training Final DQN...")
    for epoch in range(TRAINING_EPOCHS):
        epoch_seed = BASE_SEED + epoch
        print(f"\n--- Epoch {epoch+1}/{TRAINING_EPOCHS} (Seed: {epoch_seed}) ---")
        _, training_agent = run_simulation("DQN", TRAIN_STEPS_PER_EPOCH, True, epoch + 1, agent=training_agent, seed=epoch_seed)
        val_res, _ = run_simulation("DQN", EVAL_STEPS, False, 100 + epoch, agent=training_agent)
        
        if not val_res:
            raise RuntimeError(f"Validation for epoch {epoch+1} failed to produce metrics.")
            
        score = get_eval_score(val_res)
        print(f"Epoch {epoch+1} Val Score: {score:.4f}")
        train_log.append({"epoch": epoch + 1, "score": score})
        
        if score > best_eval_score:
            best_eval_score = score
            best_epoch = epoch
            training_agent.save(MODEL_PATH)
            print(f"  🥇 New best model saved (Score: {best_eval_score:.4f})")
            
        if epoch - best_epoch >= 10:
            print("--- Early stopping triggered ---")
            break
            
    pd.DataFrame(train_log).to_csv("training_log_v58.csv", index=False)
    
    print("\n📊 V58b: Final Evaluation...")
    results = {"Proposed DQN-Edge": [], "Strict Priority": [], "Random": []}
    policy_map = {"Proposed DQN-Edge": "DQN", "Strict Priority": "StrictPriority", "Random": "Random"}
    
    for i in range(20):
        run_seed = BASE_SEED + 200 + i
        print(f"\n--- Eval Run {i+1}/20 (Seed: {run_seed}) ---")
        for label, policy in policy_map.items():
            metrics, _ = run_simulation(policy, EVAL_STEPS, False, i + 1, seed=run_seed, collect_data=(i == 0))
            if metrics:
                results[label].append(metrics)
            else:
                print(f"Warning: Empty metrics for {label} on run {i+1}")
                
    final = {}
    for p, r in results.items():
        if not r:
            continue
        df = pd.DataFrame(r)
        final[p] = {k: f"{df[k].mean():.4f}±{stats.t.ppf(0.975, len(df)-1) * df[k].std() / np.sqrt(len(df)):.4f}" for k in df.columns if k != 'action'}
        
    summary_df = pd.DataFrame(final).T
    print("\n\n--- FINAL RESULTS (V58b, MEAN ± 95% CI) ---")
    print(summary_df.to_string())
    summary_df.to_excel("final_results_v58.xlsx")
    print("\n✅ Results exported to 'final_results_v58.xlsx'")
    
    if results["Proposed DQN-Edge"] and results["Strict Priority"]:
        for metric in results["Proposed DQN-Edge"][0].keys():
            if metric == 'action':
                continue
            dqn_data = [r[metric] for r in results["Proposed DQN-Edge"] if r and metric in r]
            sp_data = [r[metric] for r in results["Strict Priority"] if r and metric in r]
            n = min(len(dqn_data), len(sp_data))
            if n > 1:
                t_stat, p_val = stats.ttest_rel(dqn_data[:n], sp_data[:n])
                diff = np.array(dqn_data[:n]) - np.array(sp_data[:n])
                cohen_d = np.mean(diff) / np.std(diff, ddof=1)
                print(f"T-test for {metric} (DQN vs SP): p-value={p_val:.4e}, Cohen's d={cohen_d:.4f}")


def plot_all_figures():
    print("\n\n--- 📈 Generating Publication Plots ---")
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    datadir = os.path.join(CURRENT_DIR, 'data_v58')
    
    # Plot 1: Learning Curve
    try:
        log_df = pd.read_csv('training_log_v58.csv')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=log_df, x='epoch', y='score', marker='o', label='Validation Score', color='b', linewidth=2.5)
        best_epoch_idx = log_df['score'].idxmax()
        best_score = log_df['score'].max()
        best_epoch = log_df['epoch'][best_epoch_idx]
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model @ Epoch {best_epoch} (Score: {best_score:.4f})')
        plt.title('DQN Agent Learning Convergence', fontsize=16, fontweight='bold')
        plt.xlabel('Training Epoch')
        plt.ylabel('Composite Validation Score')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig('plot_1_learning_curve.png', dpi=300)
        plt.close()
        print("✅ Plot 1 (Learning Curve) generated.")
    except FileNotFoundError:
        print("❌ 'training_log_v58.csv' not found. Skipping Plot 1.")

    # Plot 2: Performance Bars (QoS Metrics)
    try:
        results_df = pd.read_excel('final_results_v58.xlsx', index_col=0)
        plot_data = []
        metrics_to_plot = {
            'throughput_kbps': 'Throughput (kbps)',
            'pdr_pct': 'PDR (%)',
            'plr_pct': 'PLR (%)',
            'avg_hp_delay_s': 'Avg. HP Delay (s)',
            'energy_nj_bit': 'Energy (nJ/bit)'
        }
        for policy in results_df.index:
            for metric_col, display_name in metrics_to_plot.items():
                if metric_col in results_df.columns:
                    mean_ci = results_df.loc[policy, metric_col].split('±')
                    mean = float(mean_ci[0])
                    ci = float(mean_ci[1])
                    plot_data.append([policy, display_name, mean, ci])
        plot_df = pd.DataFrame(plot_data, columns=['Policy', 'Metric', 'Mean', 'CI'])
        
        g = sns.catplot(data=plot_df, kind="bar", x="Policy", y="Mean", col="Metric", ci=None, height=5, aspect=0.9, col_wrap=3, sharey=False)
        for i, ax in enumerate(g.axes.flat):
            title = ax.get_title().split(' = ')[1]
            sub_df = plot_df[plot_df['Metric'] == title]
            ax.errorbar(x=sub_df['Policy'], y=sub_df['Mean'], yerr=sub_df['CI'], fmt='none', c='black', capsize=5, elinewidth=1.5)
        g.fig.suptitle('Performance Comparison of Scheduling Policies', y=1.03, fontsize=18, fontweight='bold')
        g.set_axis_labels("", "Mean Value")
        g.set_titles("{col_name}")
        plt.savefig('plot_2_performance_bars.png', dpi=300)
        plt.close()
        print("✅ Plot 2 (Performance Bars) generated.")
    except FileNotFoundError:
        print("❌ 'final_results_v58.xlsx' not found. Skipping Plot 2.")

    # Plot 3: Delay CDFs
    try:
        plt.figure(figsize=(10, 6))
        hp_delays = {}
        normal_delays = {}
        for policy in ['DQN', 'StrictPriority', 'Random']:
            hp_delays[policy] = np.loadtxt(os.path.join(datadir, f'{policy}_hp_delays.txt'))
            normal_delays[policy] = np.loadtxt(os.path.join(datadir, f'{policy}_normal_delays.txt'))
            
        for p, d in hp_delays.items():
            label = 'DQN-Edge (HP)' if p == 'DQN' else f'{p} (HP)'
            sns.ecdfplot(d, label=label, linewidth=2)
        for p, d in normal_delays.items():
            label = 'DQN-Edge (Normal)' if p == 'DQN' else f'{p} (Normal)'
            sns.ecdfplot(d, label=label, linestyle='--', linewidth=2)
            
        plt.title('Cumulative Distribution Function (CDF) of Packet Delays', fontsize=16, fontweight='bold')
        plt.xlabel('Delay (seconds)')
        plt.ylabel('F(x)')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':')
        plt.xlim(0, 2.5)
        plt.tight_layout()
        plt.savefig('plot_3_delay_cdf.png', dpi=300)
        plt.close()
        print("✅ Plot 3 (Delay CDF) generated.")
    except FileNotFoundError:
        print("❌ Delay log files not found in 'data_v58/'. Skipping Plot 3.")

    # Plot 4: Queue Occupancy & Average Energy over Time
    try:
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for policy in ['DQN', 'StrictPriority', 'Random']:
            q_df = pd.read_csv(os.path.join(datadir, f'{policy}_queue_log.txt'), names=['time', 'hp', 'normal'])
            e_df = pd.read_csv(os.path.join(datadir, f'{policy}_energy_log.txt'), names=['time', 'energy'])
            
            label_p = 'Proposed DQN-Edge' if policy == 'DQN' else policy
            axs[0].plot(q_df['time'], q_df['normal'], label=f'{label_p} (Normal Queue)', alpha=0.8, linewidth=1.5)
            axs[1].plot(e_df['time'], e_df['energy'], label=label_p, alpha=0.8, linewidth=1.8)
            
        axs[0].set_title('Normal Queue Buffer Congestion Over Time', fontsize=14, fontweight='bold')
        axs[0].set_ylabel('Queue Length (packets)')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, linestyle=':')
        
        axs[1].set_title('Normalized Network Energy Decay Over Time', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Simulation Time (s)')
        axs[1].set_ylabel('Residual Energy Ratio')
        axs[1].legend(loc='lower left')
        axs[1].grid(True, linestyle=':')
        
        plt.tight_layout()
        plt.savefig('plot_4_time_series.png', dpi=300)
        plt.close()
        print("✅ Plot 4 (Queue & Energy Time Series) generated.")
    except FileNotFoundError:
        print("❌ Queue/Energy log files not found in 'data_v58/'. Skipping Plot 4.")

    # Plot 5: DQN Action Distribution
    try:
        actions = np.loadtxt(os.path.join(datadir, 'DQN_actions.txt'), dtype=int)
        action_labels = ['Send HP', 'Send Normal', 'Drop Normal', 'Sleep', 'Drop HP']
        counts = pd.Series(actions).value_counts().reindex(range(len(action_labels)), fill_value=0)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=action_labels, y=counts, palette='viridis')
        plt.title('Learned Policy Action Distribution (DQN-Edge)', fontsize=16, fontweight='bold')
        plt.xlabel('Scheduling Action')
        plt.ylabel('Activation Frequency (Steps)')
        plt.grid(True, axis='y', linestyle=':')
        plt.tight_layout()
        plt.savefig('plot_5_action_dist.png', dpi=300)
        plt.close()
        print("✅ Plot 5 (Action Distribution) generated.")
    except FileNotFoundError:
        print("❌ DQN actions log not found in 'data_v58/'. Skipping Plot 5.")


if __name__ == '__main__':
    # 1. Run simulator compilation, PER-DoubleDQN training, and final 20 evaluation runs
    main_experiment() 
    
    # 2. Extract logged delays, actions, and energy metrics to generate the 5 publication figures
    plot_all_figures()
