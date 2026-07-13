import os
import sys
import subprocess
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

# --- Ensure other modules can be imported ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming train_dqn.py is in a 'training' subdirectory
RL_AGENT_ROOT = os.path.dirname(CURRENT_DIR) 
sys.path.append(os.path.join(RL_AGENT_ROOT, "networks"))

from network import DQN
from replay_buffer import ReplayBuffer
from ns3gym import ns3env

# --- Environment and Model Parameters ---
NS3_PATH = '/home/heisenberg/ns3-workspace/bake/source/ns-3.40' # IMPORTANT: Verify this path
SIM_SCRIPT = 'wsn_100dynamic' # This links to the C++ file in the scratch folder
PORT = 5555
EPISODES = 500
STEPS_PER_EPISODE = 200

# --- Module 1 Specific State and Action Sizes ---
STATE_SIZE = 5  # [NormalQ_Len, HighPriorityQ_Len, CH_Energy, Packet_Age, Sensed_Delta]
ACTION_SIZE = 4 # [Send_HP, Send_Normal, Drop_Normal, Wait]

# --- DQN Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
MEMORY_CAPACITY = 50000
TARGET_UPDATE_FREQ = 500
GRAD_CLIP = 1.0
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

class DQNAgent:
    def __init__(self):
        self.model = DQN(STATE_SIZE, ACTION_SIZE)
        self.target = DQN(STATE_SIZE, ACTION_SIZE)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        
        self.epsilon = EPS_START
        self.steps = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(ACTION_SIZE)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()
            
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        s, a, r, ns, d = self.memory.sample(BATCH_SIZE)
        
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(ns)
        d = torch.FloatTensor(d).unsqueeze(1)
        
        # Get Q-values for current states: Q(s,a)
        curr_q = self.model(s).gather(1, a)
        
        # Get max Q-values for next states from target network
        next_q_max = self.target(ns).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q-values
        target_q = r + (GAMMA * next_q_max * (1 - d))
        
        # Compute loss
        loss = F.mse_loss(curr_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target.load_state_dict(self.model.state_dict())
            
        return loss.item()
        
    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

def calculate_reward(state, prev_state):
    """
    Calculates reward based on the change in state, focusing on Module 1 objectives.
    State: [NormalQ_Len, HighPriorityQ_Len, CH_Energy, Packet_Age, Sensed_Delta]
    """
    reward = 0.0
    
    # Heavily reward sending a high-priority packet
    if state[1] < prev_state[1]: # HP Queue Length decreased
        reward += 20.0
        
    # Reward sending a normal packet
    if state[0] < prev_state[0]: # Normal Queue Length decreased
        reward += 5.0

    # Penalize stale packets in the normal queue
    # The penalty increases exponentially with age
    if state[3] > 0.5: # Packet_Age > 5 seconds (normalized)
        reward -= 2.0 * np.exp(state[3] - 0.5)

    # Penalize high congestion in the normal queue
    if state[0] > 0.8: # Normal Queue > 80% full
        reward -= 10.0

    # Small penalty for energy consumption to encourage efficiency
    energy_consumed = prev_state[2] - state[2]
    reward -= energy_consumed * 5.0

    return reward

# --- Main Execution Logic ---
def main():
    agent = DQNAgent()
    episode_rewards = deque(maxlen=100)
    best_reward = -np.inf

    print("🚀 Initializing WSN Environment for Module 1: Edge-DQN & Dual-Queue Filtering")

    for ep in range(1, EPISODES + 1):
        # Launch NS-3 simulation for each episode
        ns3_process = subprocess.Popen(
            f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT}'",
            cwd=NS3_PATH, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3) # Give time for NS-3 to start

        try:
            env = ns3env.Ns3Env(port=PORT, startSim=False)
            
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            
            for step in range(STEPS_PER_EPISODE):
                prev_state = state
                action = agent.act(state)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state = np.array(next_state, dtype=np.float32)
                
                reward = calculate_reward(next_state, prev_state)
                total_reward += reward

                agent.memory.push(state, action, reward, next_state, done)
                agent.train()
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            agent.update_epsilon()
            
            avg_reward = np.mean(episode_rewards)
            
            if avg_reward > best_reward:
                torch.save(agent.model.state_dict(), 'module1_best.pth')
                best_reward = avg_reward
                print(f"🏆 Ep {ep:3d} | Total Reward: {total_reward:6.1f} | Avg Reward: {avg_reward:6.1f} | Epsilon: {agent.epsilon:.3f} | NEW BEST!")
            else:
                 print(f"🌟 Ep {ep:3d} | Total Reward: {total_reward:6.1f} | Avg Reward: {avg_reward:6.1f} | Epsilon: {agent.epsilon:.3f}")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            env.close()
            ns3_process.kill()
            # Ensure the process is gone
            subprocess.run(f"pkill -9 -f {SIM_SCRIPT}", shell=True, check=False)
            time.sleep(1)
            
    print("\n🎯 Module 1 Training Finished. Best model saved to 'module1_best.pth'")

if __name__ == '__main__':
    main()
