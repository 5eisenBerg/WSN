import sys, os, subprocess, time, json, numpy as np, pandas as pd, torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# --- V71 Diagnostic Config ---
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(CURRENT_DIR),'networks'));from network import DQN
NS3_PATH='/home/heisenberg/ns3-workspace/bake/source/ns-3.40';SIM_SCRIPT='wsn_100dynamic';PORT=5555
MODEL_PATH=os.path.join(CURRENT_DIR,"v68_best.pth") # Use the best model from the last run
STATE_SIZE,ACTION_SIZE=6,5;EVAL_STEPS=2500;SEED=12545
ACTION_LABELS=['Send HP','Send Normal','Drop Normal','Sleep','Drop HP']

# Redefine the V68 reward function locally for analysis
HP_DELIVERY_REWARD,NORMAL_DELIVERY_REWARD,HP_DROP_PENALTY,NORMAL_DROP_PENALTY,HP_TIMEOUT_PENALTY,NORMAL_TIMEOUT_PENALTY=10.0,4.0,-12.0,-4.0,-15.0,-6.0
HP_LATENCY_PENALTY,NORMAL_LATENCY_PENALTY,ENERGY_PENALTY_WEIGHT,SLEEP_REWARD=-5.0,-1.5,400.0,1.0

def calculate_reward_v68(s, ps, info, action):
    reward = info.get('hp_sent', 0) * HP_DELIVERY_REWARD + info.get('normal_sent', 0) * NORMAL_DELIVERY_REWARD
    reward += info.get('hp_dropped', 0) * HP_DROP_PENALTY + info.get('normal_dropped', 0) * NORMAL_DROP_PENALTY
    reward += info.get('hp_timeout', 0) * HP_TIMEOUT_PENALTY + info.get('normal_timeout', 0) * NORMAL_TIMEOUT_PENALTY
    reward += s[4] * HP_LATENCY_PENALTY + s[3] * NORMAL_LATENCY_PENALTY
    energy_decay = ps[2] - s[2]
    if energy_decay > 0: reward -= energy_decay * ENERGY_PENALTY_WEIGHT
    if action == 3: reward += SLEEP_REWARD
    return reward

class DiagnosticDQNAgent:
    def __init__(self):
        self.model=DQN(STATE_SIZE,ACTION_SIZE)
    def act_and_get_q_values(self,s):
        self.model.eval()
        with torch.no_grad():
            q_values=self.model(torch.FloatTensor(s).unsqueeze(0)).flatten()
            action=q_values.argmax().item()
        return action,q_values.numpy()
    def load(self,p):
        c=torch.load(p,map_location="cpu");self.model.load_state_dict(c["model"])

def run_deep_diagnostic():
    print("--- 🔬 V71: Running Deep Diagnostic Evaluation ---")
    agent=DiagnosticDQNAgent();
    try:agent.load(MODEL_PATH);print(f"✅ Loaded model '{MODEL_PATH}'")
    except FileNotFoundError:print(f"❌ CRITICAL: Model file not found at '{MODEL_PATH}'.");return

    cmd=f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT} --run=71'";p=subprocess.Popen(cmd,cwd=NS3_PATH,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(3)
    env=None;diagnostic_log=[]
    try:
        from ns3gym import ns3env
        env=ns3env.Ns3Env(port=PORT,startSim=False);s=np.array(env.reset(),dtype=np.float32)
        bar=tqdm(range(EVAL_STEPS),desc=f"Deep Diagnostic Run")
        for step in bar:
            prev_s=s
            action,q_values=agent.act_and_get_q_values(s)
            s_next,_,d,info_str=env.step(action)
            info=json.loads(info_str if info_str else'{}');s_next=np.array(s_next,dtype=np.float32)
            
            # Log everything: state, Q-values, action, and resulting reward
            calculated_reward=calculate_reward_v68(s_next,prev_s,info,action)
            log_entry=np.concatenate([prev_s,q_values,[action,calculated_reward]])
            diagnostic_log.append(log_entry)
            
            if d:break
            s=s_next
        
        columns=['s_q_norm','s_q_hp','s_energy','s_age_norm','s_age_hp','s_neighbors']+['q_val_0','q_val_1','q_val_2','q_val_3','q_val_4']+['action','reward']
        log_df=pd.DataFrame(diagnostic_log,columns=columns)
        log_df.to_csv("deep_diagnostic_log.csv",index=False)
        print("✅ Deep diagnostic log saved to 'deep_diagnostic_log.csv'")
        return log_df
    finally:
        if env is not None:
            try:env.close()
            except:pass
        try:p.kill()
        except:pass
        subprocess.run(f"pkill -9 -f {SIM_SCRIPT}",shell=True,check=False);time.sleep(1)

def analyze_deep_diagnostic(df):
    print("\n--- 🔍 Analyzing Deep Diagnostic Log ---")
    
    # 1. Confirm the policy collapse
    action_dist=df['action'].value_counts(normalize=True)*100
    print("\nAction Distribution (%):");print(action_dist)
    if action_dist.get(0,0)==0:
        print("\n[CONFIRMED] Agent never selects 'Send HP'.")
    
    # 2. Analyze Q-Values: WHY does it always choose Action 1?
    q_cols=['q_val_0','q_val_1','q_val_2','q_val_3','q_val_4']
    df['best_action_by_q']=df[q_cols].idxmax(axis=1).str.replace('q_val_','').astype(int)
    q_value_consistency=(df['action']==df['best_action_by_q']).mean()*100
    print(f"\nQ-Value Sanity Check: Agent chose the action with the highest Q-value {q_value_consistency:.2f}% of the time.")
    
    # 3. Find states where the choice was "wrong" but Q-values insisted
    wrong_choice_df=df[(df['s_q_hp']>0)&(df['action']!=0)]
    if not wrong_choice_df.empty:
        print("\n[CRITICAL FINDING] Found states where HP queue was full, but agent chose another action.")
        print("Analyzing Q-Values for the first state where HP was ignored:")
        state_to_analyze=wrong_choice_df.iloc[0]
        print(state_to_analyze[['s_q_hp','s_q_norm','q_val_0','q_val_1','q_val_2','q_val_3','q_val_4','action']])
        
        # Plot Q-value evolution
        plt.figure(figsize=(14,7))
        for i,label in enumerate(ACTION_LABELS):
            sns.kdeplot(df[f'q_val_{i}'],label=label,linewidth=2)
        plt.title('Distribution of Predicted Q-Values For Each Action',fontsize=16,fontweight='bold')
        plt.xlabel('Predicted Q-Value');plt.legend();
        plt.savefig('deep_diagnostic_q_value_dist.png',dpi=300);plt.show()
        print("\n✅ Q-Value distribution plot generated.")

if __name__ == "__main__":
    log_df = run_deep_diagnostic()
    if log_df is not None:
        analyze_deep_diagnostic(log_df)

# new
