import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from wsn_env import WSNEnv
from network import DQN

# Load Environment and Trained DQN
env = WSNEnv()
agent = DQN(state_size=env.STATE_SIZE, action_size=env.ACTION_SIZE)
try:
    agent.load_state_dict(torch.load("module1_best.pth"))
    agent.eval()
except FileNotFoundError:
    print("Trained model not found. Run training first.")

STEPS = 200

def evaluate_policy(env, policy_type="DQN"):
    state, _ = env.reset()
    
    # Metrics to track
    total_energy_initial = sum([node['residual_energy'] for node in env.nodes])
    successful_packets = 0
    high_priority_sent = 0
    normal_packets_sent = 0
    dropped_packets = 0
    total_delay = 0.0
    step_latency = []
    step_throughput = []
    
    wrr_counter = 0
    
    for step in range(1, STEPS + 1):
        hp_q = state[1]
        normal_q = state[0]
        age = state[3]
        
        # Determine Action based on policy
        if policy_type == "DQN":
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(agent(state_t)).item()
        elif policy_type == "StrictPriority":
            if hp_q > 0: action = 0
            elif normal_q > 0: action = 1
            else: action = 3
        elif policy_type == "WeightedRoundRobin":
            if hp_q > 0 and wrr_counter < 3:
                action = 0
                wrr_counter += 1
            elif normal_q > 0:
                action = 1
                wrr_counter = 0
            else:
                action = 3
        
        # Step the environment
        next_state, _, done, truncated, _ = env.step(action)
        
        # Track throughput and queue drops
        ch_node = env.nodes[env.current_ch_id]
        if action == 0 and prev_hp_q := (state[1] * 20.0) > next_state[1] * 20.0:
            successful_packets += 1
            high_priority_sent += 1
        elif action == 1 and prev_normal_q := (state[0] * 20.0) > next_state[0] * 20.0:
            successful_packets += 1
            normal_packets_sent += 1
            total_delay += (age * 10.0) # Accumulate age/latency
        elif action == 2:
            dropped_packets += 1
            
        step_latency.append(total_delay / max(1, normal_packets_sent))
        step_throughput.append(successful_packets / step)
        
        state = next_state
        if done or truncated: break
        
    total_energy_final = sum([node['residual_energy'] for node in env.nodes])
    energy_depleted = total_energy_initial - total_energy_final
    
    # Calculate Final Metrics
    metrics = {
        "Throughput (pkts/step)": successful_packets / STEPS,
        "Average Latency (s)": total_delay / max(1, normal_packets_sent),
        "Packet Loss Rate (%)": (dropped_packets / max(1, successful_packets + dropped_packets)) * 100,
        "Energy Efficiency (nJ/bit)": (energy_depleted * 1e9) / max(1, successful_packets * 800)
    }
    return metrics, step_latency, step_throughput

# Run Evaluations
dqn_m, dqn_lat, dqn_tp = evaluate_policy(env, "DQN")
sp_m, sp_lat, sp_tp = evaluate_policy(env, "StrictPriority")
wrr_m, wrr_lat, wrr_tp = evaluate_policy(env, "WeightedRoundRobin")

# --- Export Comparative Table ---
df = pd.DataFrame([dqn_m, sp_m, wrr_m], index=["Proposed DQN-Edge", "Strict Priority", "Weighted Round Robin"])
df.to_excel("wsn_comparison_metrics.xlsx")
print("\n📊 Metric Benchmarks Exported to 'wsn_comparison_metrics.xlsx':")
print(df.to_string())

# --- Plot 1: Real-time Throughput convergence ---
plt.figure(figsize=(10, 5))
plt.plot(dqn_tp, label="Proposed DQN Edge-Scheduler", color="royalblue", linewidth=2)
plt.plot(sp_tp, label="Strict Priority (SP)", color="crimson", linestyle="--")
plt.plot(wrr_tp, label="Weighted Round Robin (WRR)", color="forestgreen", linestyle=":")
plt.title("Real-Time Network Throughput Convergence", fontsize=14, fontweight='bold')
plt.xlabel("Simulation Steps")
plt.ylabel("Throughput (Packets / Step)")
plt.legend()
plt.grid(True)
plt.savefig("throughput_comparison.png", dpi=300)
print("Saved: throughput_comparison.png")
