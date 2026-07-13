import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, "networks"))

import subprocess
import time
from collections import deque

from network import DQN
from replay_buffer import ReplayBuffer
from ns3gym import ns3env

import numpy as np
np.float = float
np.int = int

import torch
import torch.optim as optim
import torch.nn.functional as F

NS3_PATH = '/home/heisenberg/ns3-workspace/bake/source/ns-3.40'
SIM_SCRIPT = 'wsn_100dynamic' # Linked to the scaled 100-node target binary
PORT = 5555
EPISODES = 200                # Scaled up execution run size
STEPS_PER_EPISODE = 200
STATE_SIZE = 6
ACTION_SIZE = 5

BATCH_SIZE = 64
GAMMA = 0.95
LR = 5e-4
MEMORY_CAPACITY = 100000
TARGET_UPDATE_FREQ = 1000
GRAD_CLIP = 1.0
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.975              # Smooth decay tracking over 500 episodes

class EnhancedAgent:
    def __init__(self):
        self.model = DQN(STATE_SIZE, ACTION_SIZE)
        self.target = DQN(STATE_SIZE, ACTION_SIZE)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        
        self.epsilon = EPS_START
        self.steps = 0
        self.episode_rewards = deque(maxlen=100)
        
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(ACTION_SIZE)
        
        state_np = np.array(state, dtype=np.float32).flatten()
        state_t = torch.from_numpy(state_np).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()
            
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
        
    def normalize_reward(self, reward):
        delta = reward - self.reward_mean
        self.reward_mean += delta / (self.steps + 1)
        self.reward_std = np.sqrt((self.reward_std**2 * self.steps + delta**2) / (self.steps + 1))
        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-6)
        return np.clip(norm_reward, -10, 10)
        
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        s, a, r, ns, d = self.memory.sample(BATCH_SIZE)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r)
        ns = torch.FloatTensor(ns)
        d = torch.FloatTensor(d)
        
        curr_q = self.model(s).gather(1, a).squeeze(1)
        next_q = self.target(ns).max(1)[0].detach()
        target_q = r + (GAMMA * next_q * (1 - d))
        
        loss = F.mse_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.update_target()
            
        return loss.item()
        
    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

def launch_ns3():
    os.system(f"pkill -9 -f {SIM_SCRIPT}")
    time.sleep(1)
    cmd = f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT}'"
    print(f"📡 Launching Scaled 100-Node Environment: {cmd}")
    proc = subprocess.Popen(cmd, cwd=NS3_PATH, shell=True)
    time.sleep(4)
    return proc

agent = EnhancedAgent()
best_reward = float('-inf')

print(f"🚀 Initializing Dense 100-Node WSN over 200-Episode Optimization Run")

for ep in range(1, EPISODES + 1):
    ns3_process = launch_ns3()
    env = ns3env.Ns3Env(port=PORT, startSim=False)
    
    state = env.reset()
    if isinstance(state, tuple): 
        state = state[0]
    state = np.array(state, dtype=np.float32).flatten()

    print("Initial state shape:", state.shape)
    print("Initial state:", state)
    
    total_reward = 0
    episode_losses = []
    
    for step in range(STEPS_PER_EPISODE):
        action = agent.act(state)
        result = env.step(action)
        
        if len(result) == 5:
            next_state, _, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, _, done, info = result
            
        # --- SAFE BOUNDED REWARD MODEL ---
        try:
            raw_q_len = state[0] * 20.0
            raw_q_age = state[1] * 10.0
            raw_energy = state[2] * 100.0
        except IndexError:
            raw_q_len = 5.0
            raw_q_age = 0.5
            raw_energy = 50.0
        
        custom_reward = 0.0
        
        # 1. Bounded Exponential Queue-Age Timeout Penalty
        if raw_q_age > 1.2:
            age_factor = np.clip(np.exp(raw_q_age - 1.2), 0.0, 20.0)
            custom_reward -= 5.0 * age_factor
        else:
            custom_reward += 1.5 
            
        # 2. Dense Channel Queue Space Congestion Penalty
        if raw_q_len > 10:
            custom_reward -= 10.0
        elif raw_q_len < 3:
            custom_reward += 2.0
            
        # 3. Battery Life Conservation
        if raw_energy < 20:
            custom_reward -= 20.0
        else:
            custom_reward += 1.0

        shaped_reward = agent.normalize_reward(custom_reward)
        next_state = np.array(next_state, dtype=np.float32).flatten()

        if state.shape != (STATE_SIZE,) or next_state.shape != (STATE_SIZE,):
            print("=" * 60)
            print("STATE SHAPE ERROR")
            print("state shape      :", state.shape)
            print("next_state shape :", next_state.shape)
            print("state            :", state)
            print("next_state       :", next_state)
            print("=" * 60)
        
        agent.memory.push(state, action, shaped_reward, next_state, done)
        loss = agent.train()
        episode_losses.append(loss)
        
        state = next_state
        total_reward += custom_reward 
        
        if done:
            break
            
    agent.episode_rewards.append(total_reward)
    agent.update_epsilon()
    
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    avg_recent_reward = np.mean(agent.episode_rewards)
    
    if total_reward > best_reward:
        torch.save(agent.model.state_dict(), 'best_dqn_wsn_100.pth')
        best_reward = total_reward
        print(f" 🏆 NEW BEST HIGH-DENSITY PERFORMANCE: {best_reward:.1f}")
        
    print(f"🌟 Ep {ep:3d} | Reward: {total_reward:6.1f} | "
          f"MovingAvg: {avg_recent_reward:6.1f} | "
          f"Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
          
    env.close()
    ns3_process.kill()
    os.system(f"pkill -9 -f {SIM_SCRIPT}")
    time.sleep(1)

print(f"\n🎯 Scaled Optimization Run Finished. Weights stored at: best_dqn_wsn_100.pth")
