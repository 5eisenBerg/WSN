import sys, os, subprocess, time, json, numpy as np, pandas as pd, torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# --- V69 Diagnostic Config ---
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(CURRENT_DIR),'networks'));from network import DQN
NS3_PATH='/home/heisenberg/ns3-workspace/bake/source/ns-3.40';SIM_SCRIPT='wsn_100dynamic';PORT=5555
MODEL_PATH=os.path.join(CURRENT_DIR,"v68_best.pth") # Use the best model from the last run
STATE_SIZE,ACTION_SIZE=6,5;EVAL_STEPS=5000;SEED=12545 # Use a single, known seed

class DiagnosticDQNAgent:
    def __init__(self):
        self.model=DQN(STATE_SIZE,ACTION_SIZE)
        self.epsilon=0.0 # No exploration
    def act(self,s):
        self.model.eval()
        with torch.no_grad():action=self.model(torch.FloatTensor(s).unsqueeze(0)).argmax().item()
        return action
    def load(self,p):
        c=torch.load(p,map_location="cpu");self.model.load_state_dict(c["model"])

def run_diagnostic_and_log():
    print("--- 🔬 V69: Running Diagnostic Evaluation ---")
    agent=DiagnosticDQNAgent();
    try:
        agent.load(MODEL_PATH)
        print(f"✅ Successfully loaded model from '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Model file not found at '{MODEL_PATH}'. Cannot run diagnostic.");return

    cmd=f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT} --run=69'";p=subprocess.Popen(cmd,cwd=NS3_PATH,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(3)
    env=None;state_action_log=[]
    try:
        from ns3gym import ns3env
        env=ns3env.Ns3Env(port=PORT,startSim=False);s=np.array(env.reset(),dtype=np.float32)
        bar=tqdm(range(EVAL_STEPS),desc=f"Diagnostic Run")
        for step in bar:
            a=agent.act(s)
            state_action_log.append(np.concatenate([s, [a]])) # Log current state and action taken
            s_next,_,d,info_str=env.step(a)
            if d:break
            s=np.array(s_next,dtype=np.float32)
        
        # Save the log
        log_df = pd.DataFrame(state_action_log, columns=['q_norm','q_hp','energy','age_norm','age_hp','neighbors','action'])
        log_df.to_csv("diagnostic_state_action_log.csv", index=False)
        print("✅ Diagnostic log saved to 'diagnostic_state_action_log.csv'")
        return log_df
    finally:
        if env is not None:
            try:env.close()
            except:pass
        try:p.kill()
        except:pass
        subprocess.run(f"pkill -9 -f {SIM_SCRIPT}",shell=True,check=False);time.sleep(1)

def analyze_diagnostic_log(df):
    print("\n--- 🔍 Analyzing Diagnostic Log ---")
    action_counts = df['action'].value_counts(normalize=True) * 100
    print("Action Distribution (%):")
    print(action_counts)

    # Check if the agent EVER chooses to send an HP packet
    if 0 not in action_counts:
        print("\nReviewer's Suspicion Confirmed: Agent NEVER chooses 'Send HP' (Action 0).")
    
        # Find the states where sending HP would have been the obvious choice
        hp_present_df = df[df['q_hp'] > 0]
        if not hp_present_df.empty:
            print("\nThere were steps where the HP queue was NOT empty, but the agent still chose other actions.")
            print("Example states where agent ignored HP packets:")
            print(hp_present_df.head())
            
            # Plot HP Queue vs Action
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=df, x=df.index, y='q_hp', hue='action', palette='viridis')
            plt.title('HP Queue Size vs. Action Chosen by DQN')
            plt.xlabel('Simulation Step')
            plt.ylabel('HP Queue Size (Normalized)')
            plt.savefig('diagnostic_hp_queue_vs_action.png', dpi=300)
            plt.show()
            print("\n✅ Diagnostic plot 'diagnostic_hp_queue_vs_action.png' generated.")
        else:
            print("\nInteresting Finding: The HP queue was empty for the entire simulation run.")
    else:
        print("\nGood News: The agent DOES choose actions other than 'Send Normal'. The policy is adaptive.")


if __name__ == "__main__":
    log_df = run_diagnostic_and_log()
    if log_df is not None:
        analyze_diagnostic_log(log_df)

