import sys, os, subprocess, time, json, numpy as np, pandas as pd, torch, torch.optim as optim, torch.nn.functional as F
from collections import deque
from tqdm import tqdm

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(CURRENT_DIR), 'networks'))
from network import DQN

# --- Config ---
NS3_PATH = '/home/heisenberg/ns3-workspace/bake/source/ns-3.40'
SIM_SCRIPT = 'wsn_100dynamic'
PORT = 5555
MODEL_PATH = os.path.join(CURRENT_DIR, "module1_final_best.pth")

class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target = DQN(state_size, action_size)
        self.target.load_state_dict(self.model.state_dict()); self.target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
    def act(self, state):
        if np.random.rand() <= self.epsilon: return np.random.randint(4)
        with torch.no_grad(): return self.model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    def train(self):
        if len(self.memory) < 128: return
        s, a, r, ns, d = zip(*np.random.choice(np.array(self.memory, dtype=object), 128, replace=False))
        s,a,r,ns,d = torch.FloatTensor(np.array(s)),torch.LongTensor(a).unsqueeze(1),torch.FloatTensor(r).unsqueeze(1),torch.FloatTensor(np.array(ns)),torch.FloatTensor(d).unsqueeze(1)
        curr_q = self.model(s).gather(1, a)
        with torch.no_grad():
            next_a = self.model(ns).argmax(1).unsqueeze(1)
            target_q = r + 0.99 * self.target(ns).gather(1, next_a) * (1 - d)
        loss = F.smooth_l1_loss(curr_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
        if self.epsilon > 0.05: self.epsilon *= 0.995
        if len(self.memory) % 500 == 0: self.target.load_state_dict(self.model.state_dict())
    def remember(self, s, a, r, ns, d): self.memory.append((s, a, r, ns, d))

def calculate_reward(state, prev_state):
    reward = 0
    if state[1] < prev_state[1]: reward += 25
    if state[0] < prev_state[0] and state[3] < prev_state[3]: reward += 5
    if state[3] > 0.8: reward -= 10
    if state[0] > 0.8: reward -= 10
    energy_consumed = prev_state[2] - state[2]
    if energy_consumed > 0: reward -= energy_consumed * 50
    return reward

def run_simulation(policy_type, steps, train_mode=False):
    from ns3gym import ns3env
    ns3_process = subprocess.Popen(f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT}'", cwd=NS3_PATH, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        env = ns3env.Ns3Env(port=PORT, startSim=False)
        state = np.array(env.reset(), dtype=np.float32)
        agent = DoubleDQNAgent(5, 4)
        if policy_type == "DQN" and not train_mode:
            try: agent.model.load_state_dict(torch.load(MODEL_PATH)); agent.epsilon = 0.0
            except FileNotFoundError: print("⚠️ Trained model not found for evaluation. DQN will be RANDOM."); policy_type = "Random"
        
        best_avg_reward = -np.inf
        episode_rewards = deque(maxlen=100)

        loop = tqdm(range(steps), desc=f"{'Training' if train_mode else 'Evaluating'} {policy_type}")
        for step in loop:
            prev_state = state
            if policy_type == "DQN": action = agent.act(state)
            elif policy_type == "StrictPriority": action = 3; 
            if state[1] > 0: action = 0 
            elif state[0] > 0: action = 1
            else: action = np.random.randint(4)

            state, _, done, info = env.step(action)
            state = np.array(state, dtype=np.float32)
            
            if train_mode:
                reward = calculate_reward(state, prev_state)
                agent.remember(prev_state, action, reward, state, done)
                agent.train()
                episode_rewards.append(reward)
                if step % 200 == 0 and step > 0:
                    avg_reward = np.mean(episode_rewards)
                    if avg_reward > best_avg_reward: best_avg_reward = avg_reward; torch.save(agent.model.state_dict(), MODEL_PATH)
                    loop.set_postfix({"Avg Reward": f"{avg_reward:.2f}", "Epsilon": f"{agent.epsilon:.3f}"})

            if done: break
        
        return json.loads(info) if info else {}
    finally:
        try: env.close()
        except: pass
        ns3_process.kill(); subprocess.run(f"pkill -9 -f {SIM_SCRIPT}", shell=True, check=False); time.sleep(1)

if __name__ == '__main__':
    print("🛠️ Rebuilding NS-3 environment..."); build = subprocess.run(f"./ns3 build", cwd=NS3_PATH, shell=True, capture_output=True, text=True)
    if build.returncode != 0: print("❌ NS-3 Build Failed."); print(build.stderr); sys.exit(1)
    print("✅ NS-3 Build Successful.")
    
    run_simulation("DQN", steps=10000, train_mode=True)
    
    print("\n📊 Starting Final Scientific Evaluation...")
    dqn_metrics = run_simulation("DQN", steps=5000)
    sp_metrics = run_simulation("StrictPriority", steps=5000)
    random_metrics = run_simulation("Random", steps=5000)

    df = pd.DataFrame([dqn_metrics, sp_metrics, random_metrics], index=["Proposed DQN-Edge", "Strict Priority", "Random"])
    print("\n\n--- FINAL SCIENTIFIC RESULTS (VERSION 4) ---"); print(df.to_string())
    df.to_excel("final_merged_metrics.xlsx")
    print("\n✅ Results exported to 'final_merged_metrics.xlsx'")

