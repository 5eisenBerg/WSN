import sys
import os

# --- Ensure Python can locate local modules ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add both 'training' folder and 'rl-agent' root to system path
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.dirname(CURRENT_DIR))
# Also check and append 'networks' if present
networks_path = os.path.join(os.path.dirname(CURRENT_DIR), "networks")
if os.path.exists(networks_path):
    sys.path.append(networks_path)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from wsn_env import WSNEnv
from network import DQN

# --- Main Configuration ---
# Load your trained Python model.
try:
    env = WSNEnv()
    agent = DQN(state_size=env.STATE_SIZE, action_size=env.ACTION_SIZE)
    # Match the python trainer file naming output
    model_path = os.path.join(CURRENT_DIR, "module1_python_best.pth")
    if not os.path.exists(model_path):
        # Fallback to local search
        model_path = "module1_python_best.pth"
        
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    print("✅ Trained DQN model 'module1_python_best.pth' loaded successfully.")
except FileNotFoundError:
    print("⚠️ WARNING: Trained model 'module1_python_best.pth' not found. The 'DQN' policy will perform random actions.")
    agent = None 
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    agent = None

STEPS = 200

def evaluate_policy(env, policy_type="DQN"):
    """
    Runs a full evaluation for a given policy (DQN, StrictPriority, or WeightedRoundRobin)
    and returns the final metrics and per-step data for plotting.
    """
    state, _ = env.reset()
    
    # --- Metrics to track ---
    successful_packets = 0
    high_priority_sent = 0
    normal_packets_sent = 0
    dropped_packets = 0
    total_delay = 0.0
    
    # Per-step data for plotting
    step_latency_data = []
    step_throughput_data = []
    
    wrr_counter = 0
    
    # --- Initial State of the Network ---
    total_energy_initial = sum([node['residual_energy'] for node in env.nodes])

    for step in range(1, STEPS + 1):
        # Store state values *before* the action is taken
        hp_q_before = state[1] * 20.0
        normal_q_before = state[0] * 20.0
        age_before = state[3] * 10.0 # Un-normalize age
        
        # --- Determine Action based on the selected policy ---
        if policy_type == "DQN" and agent:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(agent(state_t)).item()
        elif policy_type == "StrictPriority":
            action = 3 # Default to Wait
            if hp_q_before > 0: action = 0
            elif normal_q_before > 0: action = 1
        elif policy_type == "WeightedRoundRobin":
            action = 3 # Default to Wait
            if hp_q_before > 0 and wrr_counter < 3:
                action = 0
                wrr_counter += 1
            elif normal_q_before > 0:
                action = 1
                wrr_counter = 0
        else: # Fallback for random agent if model not found
            action = env.action_space.sample()

        # --- Step the environment ---
        next_state, _, done, truncated, _ = env.step(action)
        
        # Get state values *after* the action
        hp_q_after = next_state[1] * 20.0
        normal_q_after = next_state[0] * 20.0
        
        # --- Track metric changes based on action and state change ---
        if action == 0 and hp_q_before > hp_q_after:
            successful_packets += 1
            high_priority_sent += 1
        elif action == 1 and normal_q_before > normal_q_after:
            successful_packets += 1
            normal_packets_sent += 1
            total_delay += age_before # Add the latency of the successfully sent packet
        elif action == 2 and normal_q_before > normal_q_after:
            dropped_packets += 1
            
        # Update running metrics for plotting
        current_avg_latency = total_delay / max(1, normal_packets_sent)
        current_throughput = successful_packets / step
        step_latency_data.append(current_avg_latency)
        step_throughput_data.append(current_throughput)
        
        state = next_state
        if done or truncated:
            break
            
    # --- Final calculations after the episode ends ---
    total_energy_final = sum([node['residual_energy'] for node in env.nodes])
    energy_depleted = total_energy_initial - total_energy_final
    total_generated_packets = successful_packets + dropped_packets # Approximation
    
    final_metrics = {
        "Throughput (pkts/sec)": successful_packets / STEPS,
        "Average Latency (s)": total_delay / max(1, normal_packets_sent),
        "Packet Loss Rate (%)": (dropped_packets / max(1, total_generated_packets)) * 100,
        "Energy Efficiency (nJ/bit)": (energy_depleted * 1e9) / max(1, successful_packets * env.PACKET_BITS)
    }
    return final_metrics, step_latency_data, step_throughput_data

# --- Main script execution ---
if __name__ == '__main__':
    print("📊 Running benchmark evaluations for Module 1...")
    
    # Run Evaluations for all three policies
    dqn_metrics, dqn_latency, dqn_throughput = evaluate_policy(env, "DQN")
    sp_metrics, sp_latency, sp_throughput = evaluate_policy(env, "StrictPriority")
    wrr_metrics, wrr_latency, wrr_throughput = evaluate_policy(env, "WeightedRoundRobin")

    # --- Create and Export Comparative Table ---
    df = pd.DataFrame([dqn_metrics, sp_metrics, wrr_metrics], 
                      index=["Proposed DQN-Edge", "Strict Priority", "Weighted Round Robin"])
    df.to_excel("wsn_comparison_metrics.xlsx")
    print("\n✅ Metrics benchmarks exported to 'wsn_comparison_metrics.xlsx':")
    print(df.to_string())

    # --- Plot 1: Real-time Throughput Comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_throughput, label="Proposed DQN-Edge", color="royalblue", linewidth=2)
    plt.plot(sp_throughput, label="Strict Priority (SP)", color="crimson", linestyle="--")
    plt.plot(wrr_throughput, label="Weighted Round Robin (WRR)", color="forestgreen", linestyle=":")
    plt.title("Network Throughput Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Simulation Steps", fontsize=12)
    plt.ylabel("Throughput (Packets / Step)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("throughput_comparison.png", dpi=300)
    print("\n✅ Throughput comparison plot saved to 'throughput_comparison.png'")

    # --- Plot 2: Real-time Latency Comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_latency, label="Proposed DQN-Edge", color="royalblue", linewidth=2)
    plt.plot(sp_latency, label="Strict Priority (SP)", color="crimson", linestyle="--")
    plt.plot(wrr_latency, label="Weighted Round Robin (WRR)", color="forestgreen", linestyle=":")
    plt.title("Average Packet Latency Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Simulation Steps", fontsize=12)
    plt.ylabel("Average Latency (seconds)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("latency_comparison.png", dpi=300)
    print("✅ Latency comparison plot saved to 'latency_comparison.png'")
