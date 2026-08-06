import sys, os, subprocess, time, json, numpy as np, pandas as pd, torch, torch.optim as optim, torch.nn.functional as F, random, shutil
from collections import deque
from tqdm import tqdm
import scipy.stats as stats
from per import PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# --- V76 Config: Softer Guidance & HP-Aware Evaluation ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data_v76')
FIG_DIR = os.path.join(CURRENT_DIR, 'figures_v76')
sys.path.append(os.path.join(os.path.dirname(CURRENT_DIR), 'networks'))
from network import DQN

NS3_PATH = '/home/heisenberg/ns3-workspace/bake/source/ns-3.40'
SIM_SCRIPT = 'wsn_100dynamic'
PORT = 5555
MODEL_PATH = os.path.join(CURRENT_DIR, "v76_best.pth")

STATE_SIZE, ACTION_SIZE = 6, 5
TRAINING_EPOCHS, TRAIN_STEPS_PER_EPOCH, EVAL_STEPS = 50, 5000, 5000
BASE_SEED = 12345
VALIDATION_SEEDS = [99991, 99992, 99993, 99994, 99995]

# V76: Final Reward Function with Softer, Urgent-Only Guidance
HP_DELIVERY_REWARD = 25.0
NORMAL_DELIVERY_REWARD = 5.0
HP_DROP_PENALTY = -40.0
NORMAL_DROP_PENALTY = -8.0
HP_TIMEOUT_PENALTY = -80.0
NORMAL_TIMEOUT_PENALTY = -10.0
HP_LATENCY_PENALTY = -40.0
NORMAL_LATENCY_PENALTY = -2.0
ENERGY_PENALTY_WEIGHT = 250.0
SLEEP_REWARD = 0.5
# --- V76 REWARD SHAPING (SOFT GUIDANCE) ---
HP_IGNORE_PENALTY = -8.0
HP_SERVICE_REWARD = 6.0
HP_URGENT_AGE_THRESHOLD = 0.55

EPSILON_DECAY = 0.99995
WARMUP_STEPS = 4000

POLICIES = {"Proposed DQN-Edge": "DQN", "Strict Priority": "StrictPriority", "Random": "Random"}
ACTION_LABELS = ['Send HP', 'Send Normal', 'Drop Normal', 'Sleep', 'Drop HP']
METRICS_FOR_STATS = {'throughput_kbps': 'Throughput (kbps)', 'pdr_pct': 'PDR (%)', 'avg_hp_delay_s': 'Avg. HP Delay (s)', 'energy_nj_bit': 'Energy (nJ/bit)'}

torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PER_DoubleDQNAgent:
    def __init__(self, tau=0.005, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.model = DQN(STATE_SIZE, ACTION_SIZE)
        self.target = DQN(STATE_SIZE, ACTION_SIZE)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = PrioritizedReplayBuffer(100000, alpha)
        self.epsilon = 1.0
        self.train_steps = 0
        self.tau = tau
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
        if len(self.memory) < WARMUP_STEPS:
            return
        self.model.train()
        beta = min(1.0, self.beta_start + self.train_steps * (1.0 - self.beta_start) / self.beta_frames)
        samples, idx, w = self.memory.sample(128, beta)
        s, a, r, ns, d = zip(*samples)
        s = torch.FloatTensor(np.array(s)); a = torch.LongTensor(a).unsqueeze(1); r = torch.FloatTensor(r).unsqueeze(1); ns = torch.FloatTensor(np.array(ns)); d = torch.FloatTensor(d).unsqueeze(1); w = torch.FloatTensor(w).unsqueeze(1)
        curr_q = self.model(s).gather(1, a)
        with torch.no_grad():
            next_a = self.model(ns).argmax(1).unsqueeze(1)
            target_q = r + 0.99 * self.target(ns).gather(1, next_a) * (1 - d)
        td_error = torch.abs(target_q - curr_q)
        loss = (w * F.smooth_l1_loss(curr_q, target_q, reduction='none')).mean()
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step(); self.scheduler.step()
        prios = td_error.detach().cpu().numpy().flatten() + 1e-5
        self.memory.update_priorities(idx, prios)
        if len(self.memory) > WARMUP_STEPS:
            self.epsilon = max(0.05, self.epsilon * EPSILON_DECAY)
        self._soft_update_target()
        self.train_steps += 1

    def _soft_update_target(self):
        for tp, lp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def remember(self, s, a, r, ns, d):
        self.memory.add(s, a, r, ns, d)

    def save(self, p):
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "epsilon": self.epsilon, "step": self.train_steps}, p)

    def load(self, p):
        c = torch.load(p, map_location="cpu")
        self.model.load_state_dict(c["model"]); self.optimizer.load_state_dict(c["optimizer"]); self.scheduler.load_state_dict(c["scheduler"])
        self.epsilon = c["epsilon"]; self.train_steps = c["step"]; self.target.load_state_dict(self.model.state_dict())

def calculate_reward_v76(s, ps, info):
    reward = info.get('hp_sent', 0) * HP_DELIVERY_REWARD + info.get('normal_sent', 0) * NORMAL_DELIVERY_REWARD
    reward += info.get('hp_dropped', 0) * HP_DROP_PENALTY + info.get('normal_dropped', 0) * NORMAL_DROP_PENALTY
    reward += info.get('hp_timeout', 0) * HP_TIMEOUT_PENALTY + info.get('normal_timeout', 0) * NORMAL_TIMEOUT_PENALTY
    reward += (s[4] ** 2) * HP_LATENCY_PENALTY + (s[3] ** 2) * NORMAL_LATENCY_PENALTY
    hp_queue_has_packets = ps[1] > 0
    hp_age_prev = ps[4]
    action_taken = info.get('action')
    if hp_queue_has_packets:
        if action_taken == 0:
            reward += HP_SERVICE_REWARD
        elif hp_age_prev >= HP_URGENT_AGE_THRESHOLD:
            reward += HP_IGNORE_PENALTY
    energy_decay = ps[2] - s[2]
    if energy_decay > 0:
        reward -= energy_decay * ENERGY_PENALTY_WEIGHT
    if action_taken == 3:
        reward += SLEEP_REWARD
    return np.tanh(reward / 100.0)

def run_simulation(policy, steps, train_mode=False, run_id=1, agent=None, seed=BASE_SEED, collect_data=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from ns3gym import ns3env
    cmd = f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT} --run={run_id}'"
    p = subprocess.Popen(cmd, cwd=NS3_PATH, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    env = None; original_epsilon = None; actions_log = []; rewards_log = []
    try:
        env = ns3env.Ns3Env(port=PORT, startSim=False)
        s = np.array(env.reset(), dtype=np.float32)
        if agent is None:
            agent = PER_DoubleDQNAgent()
            if policy == "DQN" and not train_mode:
                try:
                    agent.load(MODEL_PATH); agent.epsilon = 0.0
                except FileNotFoundError:
                    print("⚠️ Model not found. DQN is RANDOM."); policy = "Random"
        elif policy == "DQN" and not train_mode:
            original_epsilon = agent.epsilon; agent.epsilon = 0.0
        final_info = {}
        bar = tqdm(range(steps), desc=f"{'Eval' if not train_mode else 'Train'} {policy} Run {run_id}")
        for step in bar:
            ps = s.copy(); a = np.random.randint(ACTION_SIZE)
            if not (train_mode and len(agent.memory) < WARMUP_STEPS):
                if policy == "DQN": a = agent.act(s)
                elif policy == "StrictPriority":
                    if s[1] > 0: a = 0
                    elif s[0] > 0: a = 1
                    else: a = 3
            if collect_data: actions_log.append(a)
            s_next, _, d, info_str = env.step(a)
            info = json.loads(info_str if info_str else '{}'); info['action'] = a; s_next = np.array(s_next, dtype=np.float32)
            if train_mode:
                r = 0 if d else calculate_reward_v76(s_next, ps, info)
                rewards_log.append(r); agent.remember(s, a, r, s_next, d); agent.train()
            s = s_next
            if d:
                if "throughput_kbps" not in info: raise RuntimeError("Simulator did not return final metrics.")
                final_info = info; break
        if collect_data:
            os.makedirs(DATA_DIR, exist_ok=True)
            if train_mode: np.savetxt(os.path.join(DATA_DIR, f'reward_log_epoch{run_id}.txt'), rewards_log)
            for fname in ['hp_delays.txt', 'normal_delays.txt', 'queue_log.txt', 'energy_log.txt']:
                src = os.path.join(NS3_PATH, fname); dst = os.path.join(DATA_DIR, f'{policy}_{fname}')
                if os.path.exists(src): shutil.copy(src, dst)
            np.savetxt(os.path.join(DATA_DIR, f'{policy}_actions.txt'), actions_log, fmt='%d')
        return (final_info, agent)
    finally:
        if original_epsilon is not None and agent is not None: agent.epsilon = original_epsilon
        if env is not None:
            try: env.close()
            except: pass
        try: p.kill()
        except: pass
        subprocess.run(f"pkill -9 -f {SIM_SCRIPT}", shell=True, check=False); time.sleep(1)

def get_eval_score(res):
    pdr = res.get('pdr_pct', 0.0)
    plr = res.get('plr_pct', 100.0)
    hd = res.get('avg_hp_delay_s', 2.0)
    hp_generated = res.get('hp_generated', 0.0)
    hp_delivered = res.get('hp_delivered', 0.0)
    hp_delivery_ratio = res.get('hp_delivery_ratio_pct', 0.0)
    hp_delay_component = 0.0 if hp_delivered <= 0 else (1.0 - min(hd / 1.0, 1.0))
    score = (0.40 * (pdr / 100.0) + 0.10 * (1.0 - plr / 100.0) + 0.35 * (hp_delivery_ratio / 100.0) + 0.15 * hp_delay_component)
    if hp_generated > 0 and hp_delivered == 0:
        score -= 0.25
    return np.clip(score, 0.0, 1.0)

def main_experiment():
    print("🛠️ V76: Rebuilding NS-3..."); build = subprocess.run(f"./ns3 build", cwd=NS3_PATH, shell=True, capture_output=True, text=True)
    if build.returncode != 0: print(f"❌ Build Fail:\n{build.stderr}"); sys.exit(1)
    print("✅ Build OK.")
    if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    training_agent = PER_DoubleDQNAgent(); best_eval_score = float("-inf"); best_epoch = 0; train_log = []
    print("\n🧠 V76: Training Final DQN with Soft Policy Guidance...")
    for epoch in range(TRAINING_EPOCHS):
        epoch_seed = BASE_SEED + epoch; print(f"\n--- Epoch {epoch + 1}/{TRAINING_EPOCHS} (Seed: {epoch_seed}) ---")
        _, training_agent = run_simulation("DQN", TRAIN_STEPS_PER_EPOCH, True, epoch + 1, agent=training_agent, seed=epoch_seed, collect_data=(epoch == 0))
        val_scores = []
        for i, val_seed in enumerate(VALIDATION_SEEDS):
            val_res, _ = run_simulation("DQN", EVAL_STEPS, False, 1000 + epoch * 10 + i, agent=training_agent, seed=val_seed)
            if val_res: val_scores.append(get_eval_score(val_res))
        if not val_scores: raise RuntimeError(f"Validation failed for epoch {epoch + 1}.")
        mean_score = np.mean(val_scores)
        print(f"Epoch {epoch + 1} Mean Val Score: {mean_score:.4f} (across {len(VALIDATION_SEEDS)} seeds)")
        train_log.append({"epoch": epoch + 1, "score": mean_score, "epsilon": training_agent.epsilon, "lr": training_agent.optimizer.param_groups[0]['lr'], "buffer": len(training_agent.memory)})
        if mean_score > best_eval_score:
            best_eval_score = mean_score; best_epoch = epoch; training_agent.save(MODEL_PATH)
            print(f"  🥇 New best model saved (Score: {best_eval_score:.4f})")
        if epoch - best_epoch >= 15: print("--- Early stopping triggered ---"); break
    pd.DataFrame(train_log).to_csv("training_log_v76.csv", index=False)
    print("\n📊 V76: Final Evaluation..."); raw_results = []
    for i in range(20):
        run_seed = BASE_SEED + 200 + i; print(f"\n--- Eval Run {i + 1}/20 (Seed: {run_seed}) ---")
        for label, policy in POLICIES.items():
            metrics, _ = run_simulation(policy, EVAL_STEPS, False, i + 1, seed=run_seed, collect_data=(i == 0))
            if metrics: metrics['Policy'] = label; metrics['run'] = i + 1; raw_results.append(metrics)
            else: print(f"Warning: Empty metrics for {label} on run {i + 1}")
    raw_df = pd.DataFrame(raw_results); raw_df.to_csv('raw_results_v76.csv', index=False)
    summary_data = []
    for policy, group in raw_df.groupby('Policy'):
        entry = {'Policy': policy}
        for col in group.columns:
            if col in ["Policy", "run", "action"]: continue
            mean = group[col].mean(); ci = stats.t.ppf(0.975, len(group) - 1) * group[col].std(ddof=1) / np.sqrt(len(group))
            entry[col] = f"{mean:.4f}±{ci:.4f}"
        summary_data.append(entry)
    summary_df = pd.DataFrame(summary_data).set_index('Policy'); print("\n\n--- FINAL RESULTS (V76, MEAN ± 95% CI) ---"); print(summary_df.to_string())
    summary_df.to_excel("final_results_v76.xlsx"); print("\n✅ Raw and summary results exported.")

def plot_all_figures():
    print("\n\n--- 📈 Generating Publication Plots ---"); sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2); os.makedirs(FIG_DIR, exist_ok=True)
    try:
        raw_df = pd.read_csv('raw_results_v76.csv'); log_df = pd.read_csv('training_log_v76.csv')
    except FileNotFoundError: print("❌ Data files not found. Cannot generate plots."); return
    try:
        fig, ax1 = plt.subplots(figsize=(12, 7)); ax2 = ax1.twinx(); ax1.plot(log_df['epoch'], log_df['score'].rolling(5, min_periods=1, center=True).mean(), 'b-', label='Validation Score (5-Epoch Avg)', linewidth=3); ax2.plot(log_df['epoch'], log_df['epsilon'], 'r--', label='Epsilon Decay', alpha=0.6); ax1.set_xlabel('Training Epoch'); ax1.set_ylabel('Validation Score', color='b'); ax2.set_ylabel('Epsilon', color='r'); best_idx = log_df['score'].idxmax(); best_score = log_df['score'].max(); best_epoch = log_df['epoch'][best_idx]; ax1.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model @ Epoch {best_epoch}'); ax1.scatter(best_epoch, best_score, color='g', s=150, zorder=5); plt.title('Agent Training Progression', fontsize=18, fontweight='bold'); fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, 'fig_1_learning_curve.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 1: Learning Curve")
        rewards = np.loadtxt(os.path.join(DATA_DIR, 'reward_log_epoch1.txt')); reward_df = pd.DataFrame({'step': range(len(rewards)), 'reward': rewards}); reward_df['reward_smooth'] = reward_df['reward'].rolling(window=100, min_periods=1).mean(); plt.figure(figsize=(12, 7)); plt.plot(reward_df['step'], reward_df['reward_smooth']); plt.title('Per-Step Reward (First Training Epoch)', fontsize=18, fontweight='bold'); plt.xlabel('Training Step'); plt.ylabel('Smoothed Reward (100-step avg)'); plt.savefig(os.path.join(FIG_DIR, 'fig_2_reward_curve.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 2: Reward Curve")
    except Exception as e: print(f"❌ Error in training plots: {e}")
    df_melted = raw_df.melt(id_vars='Policy', value_vars=METRICS_FOR_STATS.keys(), var_name='Metric', value_name='Value'); df_melted['Metric'] = df_melted['Metric'].map(METRICS_FOR_STATS)
    g = sns.catplot(data=df_melted, kind='box', x='Policy', y='Value', col='Metric', col_wrap=2, height=5, aspect=1.3, sharey=False, order=list(POLICIES.keys())); g.fig.suptitle('Distribution of Performance Metrics (20 Runs)', y=1.03, fontsize=18, fontweight='bold'); g.set_axis_labels("", "Value"); g.set_titles("{col_name}"); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(FIG_DIR, 'fig_3_boxplots.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 3: Boxplots")
    g = sns.catplot(data=df_melted, kind='bar', x='Policy', y='Value', col='Metric', col_wrap=2, height=5, aspect=1.3, sharey=False, order=list(POLICIES.keys()), errorbar=None); g.fig.suptitle('Mean Performance Comparison (95% CI)', y=1.03, fontsize=18, fontweight='bold'); g.set_axis_labels("", "Mean Value"); g.set_titles("{col_name}");
    for ax, metric in zip(g.axes.flat, METRICS_FOR_STATS.values()):
        sub_df = df_melted[df_melted['Metric'] == metric]; means = sub_df.groupby('Policy')['Value'].mean().loc[list(POLICIES.keys())]; cis = sub_df.groupby('Policy')['Value'].apply(lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))).loc[list(POLICIES.keys())]
        ax.errorbar(x=range(len(means)), y=means, yerr=cis, fmt='none', c='black', capsize=5)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(FIG_DIR, 'fig_4_bar_charts.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 4: Bar Charts")
    plt.figure(figsize=(12, 7))
    for label, p_short in POLICIES.items():
        try:
            path_hp = os.path.join(DATA_DIR, f'{p_short}_hp_delays.txt')
            if os.path.exists(path_hp) and os.path.getsize(path_hp) > 0:
                hp_delays = np.loadtxt(path_hp, ndmin=1)
                sns.ecdfplot(hp_delays, label=f'{label} (HP)', linewidth=3)
            else:
                print(f"⚠️ Empty or missing HP delay log for {label}, skipping HP CDF.")
            path_normal = os.path.join(DATA_DIR, f'{p_short}_normal_delays.txt')
            if os.path.exists(path_normal) and os.path.getsize(path_normal) > 0:
                normal_delays = np.loadtxt(path_normal, ndmin=1)
                sns.ecdfplot(normal_delays, label=f'{label} (Normal)', linestyle='--', linewidth=3)
            else:
                print(f"⚠️ Empty or missing normal delay log for {label}, skipping normal CDF.")
        except FileNotFoundError:
            print(f"⚠️ No delay logs for {label}, skipping in CDF plot.")
    plt.title('CDF of Packet End-to-End Delays', fontsize=18, fontweight='bold'); plt.xlabel('Delay (s)'); plt.ylabel('Probability (CDF)'); plt.legend(); plt.grid(True, linestyle=':'); plt.xlim(left=0, right=2.5); plt.savefig(os.path.join(FIG_DIR, 'fig_5_delay_cdf.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 5: Delay CDF")
    all_q_data = []
    for label, p_short in POLICIES.items():
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f'{p_short}_queue_log.txt'), names=['time', 'hp', 'normal']); df['Policy'] = label; all_q_data.append(df)
        except FileNotFoundError: continue
    if all_q_data:
        q_df = pd.concat(all_q_data); plt.figure(figsize=(12, 7)); sns.violinplot(data=q_df, x='Policy', y='normal', order=list(POLICIES.keys()), inner='box', cut=0); plt.title('Distribution of Normal Queue Occupancy', fontsize=18, fontweight='bold'); plt.xlabel('Policy'); plt.ylabel('Queue Length (packets)'); plt.savefig(os.path.join(FIG_DIR, 'fig_6_queue_violin.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 6: Queue Violin Plot")
    try:
        actions = np.loadtxt(os.path.join(DATA_DIR, 'DQN_actions.txt'), dtype=int); counts = pd.Series(actions).value_counts().reindex(range(len(ACTION_LABELS)), fill_value=0); percentages = counts / counts.sum() * 100
        plt.figure(figsize=(10, 6)); ax = sns.barplot(x=ACTION_LABELS, y=percentages, palette='viridis'); ax.bar_label(ax.containers[0], fmt='%.1f%%'); plt.title('DQN Agent Learned Policy', fontsize=18, fontweight='bold'); plt.xlabel('Action'); plt.ylabel('Policy Usage (%)'); plt.ylim(0, 1.1 * percentages.max()); plt.savefig(os.path.join(FIG_DIR, 'fig_7_action_dist.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 7: Action Distribution")
    except FileNotFoundError: print("❌ Action log file not found. Skipping plot.")
    summary_df = raw_df.groupby('Policy').mean().reset_index(); metrics_radar = ['pdr_pct', 'throughput_kbps', 'avg_hp_delay_s', 'energy_nj_bit']; labels_radar = ['PDR (%) ↑', 'Throughput (kbps) ↑', 'HP Delay (s) ↓', 'Energy (nJ/bit) ↓']
    df_norm = summary_df.copy();
    for metric in metrics_radar:
        min_val, max_val = df_norm[metric].min(), df_norm[metric].max();
        if metric in ['avg_hp_delay_s', 'energy_nj_bit']: df_norm[metric] = (max_val - df_norm[metric]) / (max_val - min_val) if (max_val - min_val) != 0 else 0
        else: df_norm[metric] = (df_norm[metric] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
    angles = [n / float(len(labels_radar)) * 2 * pi for n in range(len(labels_radar))]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True)); ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1); plt.xticks(angles[:-1], labels_radar); ax.set_rlabel_position(0); plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7); plt.ylim(0, 1)
    for i, row in df_norm.iterrows():
        if row['Policy'] not in POLICIES.keys(): continue
        data = row[metrics_radar].values.flatten().tolist(); data += data[:1]; ax.plot(angles, data, linewidth=2, linestyle='solid', label=row['Policy']); ax.fill(angles, data, alpha=0.25)
    plt.title('Multi-Objective Performance Radar Chart', size=20, color='black', y=1.1); plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)); plt.savefig(os.path.join(FIG_DIR, 'fig_8_radar_chart.png'), dpi=300, bbox_inches="tight"); plt.close(); print("✅ Fig 8: Radar Chart")

def generate_stats_table():
    print("\n\n--- 📋 Statistical Significance Table ---")
    try:
        raw_df = pd.read_csv('raw_results_v76.csv')
        dqn_data = raw_df[raw_df['Policy'] == 'Proposed DQN-Edge']; sp_data = raw_df[raw_df['Policy'] == 'Strict Priority']
        stats_results = []
        for metric, name in METRICS_FOR_STATS.items():
            _, p_norm_dqn = stats.shapiro(dqn_data[metric]); _, p_norm_sp = stats.shapiro(sp_data[metric])
            test_used = "Welch's t-test" if p_norm_dqn > 0.05 and p_norm_sp > 0.05 else "Mann-Whitney U"
            p_val = stats.ttest_ind(dqn_data[metric], sp_data[metric], equal_var=False).pvalue if test_used == "Welch's t-test" else stats.mannwhitneyu(dqn_data[metric], sp_data[metric], alternative='two-sided').pvalue
            n1, n2 = len(dqn_data), len(sp_data); s1, s2 = dqn_data[metric].std(ddof=1), sp_data[metric].std(ddof=1); pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            cohen_d = (dqn_data[metric].mean() - sp_data[metric].mean()) / pooled_std if pooled_std > 0 else 0
            stats_results.append({'Metric': name, 'p-value (vs. SP)': f"{p_val:.3e}", "Cohen's d": f"{cohen_d:.4f}", "Test Used": test_used})
        stats_df = pd.DataFrame(stats_results).set_index('Metric'); print(stats_df.to_markdown()); print("\n" + stats_df.to_latex())
    except Exception as e:
        print(f"❌ Could not generate Stats Table: {e}")

if __name__ == "__main__":
    main_experiment()
    plot_all_figures()
    generate_stats_table()
