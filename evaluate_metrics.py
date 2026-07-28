import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Robust Import Handling ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, 'networks'))

try:
    from wsn_env import WSNEnv
    from network import DQN
    print("✅ Environment and Network modules loaded successfully.")
except ImportError:
    print("❌ CRITICAL ERROR: Could not find 'wsn_env.py' or 'network.py'.")
    print("   Please ensure 'wsn_env.py' is in the 'training/' directory and 'network.py' is in 'networks/'.")
    sys.exit(1)

# --- Main Configuration ---
LONG_RUN_STEPS = 5000 # Run for a long time to capture node death

# --- Load Trained DQN Model ---
try:
    env = WSNEnv()
    agent = DQN(state_size=env.STATE_SIZE, action_size=env.ACTION_SIZE)
    model_path = os.path.join(CURRENT_DIR, "module1_python_best.pth")
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    print(f"✅ Trained DQN model '{model_path}' loaded.")
except FileNotFoundError:
    print(f"⚠️ WARNING: Trained model '{model_path}' not found. The 'DQN' policy will be RANDOM.")
    agent = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    agent = None

def run_long_evaluation(env, policy_type="DQN"):
    """
    Runs a long-duration evaluation to capture lifecycle and performance metrics.
    """
    state, _ = env.reset()
    
    # --- Per-Step Tracking ---
    alive_nodes_over_time = []
    avg_residual_energy_over_time = []
    
    # --- Cumulative Metrics ---
    successful_packets = 0
    dropped_packets = 0
    total_delay = 0.0
    normal_packets_sent = 0
    wrr_counter = 0

    # --- Lifecycle Metrics ---
    fnd_step = -1  # First Node Dies
    hna_step = -1  # Half Nodes Alive
    lnd_step = -1  # Last Node Dies
    
    total_energy_initial = sum([node['residual_energy'] for node in env.nodes])

    progress_bar = tqdm(range(1, LONG_RUN_STEPS + 1), desc=f"Evaluating {policy_type}")
    for step in progress_bar:
        # --- Policy Decision ---
        hp_q_before = state[1] * 20.0
        normal_q_before = state[0] * 20.0
        age_before = state[3] * 10.0
        
        if policy_type == "DQN" and agent:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): action = torch.argmax(agent(state_t)).item()
        elif policy_type == "StrictPriority":
            action = 3; 
            if hp_q_before > 0: action = 0
            elif normal_q_before > 0: action = 1
        elif policy_type == "WeightedRoundRobin":
            action = 3
            if hp_q_before > 0 and wrr_counter < 3: action = 0; wrr_counter += 1
            elif normal_q_before > 0: action = 1; wrr_counter = 0
        else: action = env.action_space.sample()

        # --- Environment Step ---
        next_state, _, _, _, _ = env.step(action)
        
        # --- Track Performance Metrics ---
        hp_q_after, normal_q_after = next_state[1] * 20.0, next_state[0] * 20.0
        if action == 0 and hp_q_before > hp_q_after: successful_packets += 1
        elif action == 1 and normal_q_before > normal_q_after:
            successful_packets += 1; normal_packets_sent += 1; total_delay += age_before
        elif action == 2 and normal_q_before > normal_q_after: dropped_packets += 1
            
        # --- Track Lifecycle & Energy Metrics ---
        alive_nodes = sum(1 for node in env.nodes if node['residual_energy'] > 0)
        alive_nodes_over_time.append(alive_nodes)
        
        avg_energy = np.mean([node['residual_energy'] for node in env.nodes])
        avg_residual_energy_over_time.append(avg_energy)

        if alive_nodes < env.NUM_NODES and fnd_step == -1: fnd_step = step
        if alive_nodes <= (env.NUM_NODES / 2) and hna_step == -1: hna_step = step
        if alive_nodes == 0:
            if lnd_step == -1: lnd_step = step
            break # End simulation if all nodes are dead
            
        state = next_state

    # --- Final Metric Calculations ---
    total_energy_final = sum([node['residual_energy'] for node in env.nodes])
    energy_consumed = total_energy_initial - total_energy_final
    total_generated = successful_packets + dropped_packets
    
    final_metrics = {
        "FND (step)": fnd_step,
        "HNA (step)": hna_step,
        "LND (step)": lnd_step if lnd_step != -1 else ">" + str(LONG_RUN_STEPS),
        "Throughput (pkts/sec)": successful_packets / step,
        "Packet Delivery Ratio (%)": (successful_packets / max(1, total_generated)) * 100,
        "Average Latency (s)": total_delay / max(1, normal_packets_sent),
        "Energy Efficiency (nJ/bit)": (energy_consumed * 1e9) / max(1, successful_packets * env.PACKET_BITS)
    }
    
    return final_metrics, alive_nodes_over_time, avg_residual_energy_over_time

if __name__ == '__main__':
    print("📊 Starting Final Benchmark Evaluation...")
    
    policies_to_run = ["Proposed DQN-Edge", "Strict Priority", "Weighted Round Robin"]
    results = {}
    
    for policy in policies_to_run:
        metrics, alive_nodes, avg_energy = run_long_evaluation(env, policy.replace("Proposed DQN-Edge", "DQN"))
        results[policy] = {"metrics": metrics, "alive_nodes": alive_nodes, "avg_energy": avg_energy}

    # --- Create and Export Summary Table ---
    summary_df = pd.DataFrame({p: r["metrics"] for p, r in results.items()}).T
    excel_path = os.path.join(CURRENT_DIR, "final_metrics_comparison.xlsx")
    summary_df.to_excel(excel_path)
    print(f"\n\n✅ Final metrics exported to '{excel_path}':")
    print(summary_df.to_string())

    # --- Generate Plots ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Network Lifetime
    plt.figure(figsize=(12, 7))
    for policy, data in results.items():
        plt.plot(data["alive_nodes"], label=policy, linewidth=2)
    plt.title("Network Lifetime Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Simulation Steps", fontsize=12)
    plt.ylabel("Number of Alive Nodes", fontsize=12)
    plt.legend()
    plt.grid(True)
    lifetime_path = os.path.join(CURRENT_DIR, "network_lifetime_comparison.png")
    plt.savefig(lifetime_path, dpi=300); print(f"\n✅ Lifetime plot saved to '{lifetime_path}'")

    # Plot 2: Average Residual Energy
    plt.figure(figsize=(12, 7))
    for policy, data in results.items():
        plt.plot(data["avg_energy"], label=policy, linewidth=2)
    plt.title("Average Residual Energy Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Simulation Steps", fontsize=12)
    plt.ylabel("Average Energy per Node (Joules)", fontsize=12)
    plt.legend()
    plt.grid(True)
    energy_path = os.path.join(CURRENT_DIR, "residual_energy_comparison.png")
    plt.savefig(energy_path, dpi=300); print(f"✅ Energy plot saved to '{energy_path}'")

