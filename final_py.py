import sys, os, subprocess, time, json, numpy as np, pandas as pd, torch, torch.optim as optim, torch.nn.functional as F, random
from collections import deque
from tqdm import tqdm
import scipy.stats as stats
from per import PrioritizedReplayBuffer

# --- V53 Config ---
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__));sys.path.append(os.path.join(os.path.dirname(CURRENT_DIR),'networks'));from network import DQN
NS3_PATH='/home/heisenberg/ns3-workspace/bake/source/ns-3.40';SIM_SCRIPT='wsn_100dynamic';PORT=5555;MODEL_PATH=os.path.join(CURRENT_DIR,"v53_best.pth")
STATE_SIZE,ACTION_SIZE=7,6;TRAINING_EPOCHS,TRAIN_STEPS_PER_EPOCH,EVAL_STEPS,BASE_SEED=40,2500,5000,12345

random.seed(BASE_SEED);np.random.seed(BASE_SEED);torch.manual_seed(BASE_SEED)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

class PER_DoubleDQNAgent:
    def __init__(self, tau=0.005, warmup=1000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.model=DQN(STATE_SIZE,ACTION_SIZE);self.target=DQN(STATE_SIZE,ACTION_SIZE);self.target.load_state_dict(self.model.state_dict());self.target.eval()
        self.optimizer=optim.Adam(self.model.parameters(),lr=1e-4);self.memory=PrioritizedReplayBuffer(100000,alpha);self.epsilon=1.0;self.train_steps=0;self.tau=tau;self.warmup=warmup
        self.beta_start=beta_start;self.beta_frames=beta_frames;self.scheduler=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=TRAINING_EPOCHS*TRAIN_STEPS_PER_EPOCH)
    def act(self,s):
        if np.random.rand()<=self.epsilon:return np.random.randint(ACTION_SIZE)
        self.model.eval();
        with torch.no_grad():action=self.model(torch.FloatTensor(s).unsqueeze(0)).argmax().item()
        self.model.train();return action
    def train(self):
        if len(self.memory)<self.warmup:return
        self.model.train();beta=min(1.0,self.beta_start+self.train_steps*(1.0-self.beta_start)/self.beta_frames)
        samples,idx,w=self.memory.sample(128,beta);s,a,r,ns,d=zip(*samples)
        s=torch.FloatTensor(np.array(s));a=torch.LongTensor(a).unsqueeze(1);r=torch.FloatTensor(r).unsqueeze(1)
        ns=torch.FloatTensor(np.array(ns));d=torch.FloatTensor(d).unsqueeze(1);w=torch.FloatTensor(w).unsqueeze(1)
        curr_q=self.model(s).gather(1,a);
        with torch.no_grad():next_a=self.model(ns).argmax(1).unsqueeze(1);target_q=r+0.99*self.target(ns).gather(1,next_a)*(1-d)
        td_error=torch.abs(target_q-curr_q);loss=(w*F.smooth_l1_loss(curr_q,target_q,reduction='none')).mean()
        self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step();self.scheduler.step()
        prios=td_error.detach().cpu().numpy().flatten()+1e-5;self.memory.update_priorities(idx,prios)
        if len(self.memory)>self.warmup:self.epsilon=max(0.05,self.epsilon*0.9995)
        self._soft_update_target();self.train_steps+=1
    def _soft_update_target(self):
        for tp,lp in zip(self.target.parameters(),self.model.parameters()):tp.data.copy_(self.tau*lp.data+(1.0-self.tau)*tp.data)
        self.target.eval()
    def remember(self,s,a,r,ns,d):self.memory.add(s,a,r,ns,d)
    def save(self,p):torch.save({"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict(),"epsilon":self.epsilon,"step":self.train_steps},p)
    def load(self,p):
        c=torch.load(p,map_location="cpu");self.model.load_state_dict(c["model"]);self.optimizer.load_state_dict(c["optimizer"]);self.scheduler.load_state_dict(c["scheduler"])
        self.epsilon=c["epsilon"];self.train_steps=c["step"];self.target.load_state_dict(self.model.state_dict())

def calculate_reward(s, ps, info):
    reward = 0
    reward += info.get('hp_sent', 0) * 2.0
    reward += info.get('normal_sent', 0) * 1.0
    reward -= info.get('hp_dropped', 0) * 10.0
    reward -= info.get('normal_dropped', 0) * 5.0
    reward -= info.get('hp_timeout', 0) * 12.0
    reward -= info.get('normal_timeout', 0) * 6.0
    reward -= (ps[2] - s[2]) * 2000
    reward -= s[3] * 0.5
    reward -= s[4] * 1.0
    return reward

def get_eval_score(res):
    t,p,plr,nd,hd,e=res.get('throughput_kbps',0),res.get('pdr_pct',0),res.get('plr_pct',0),res.get('avg_normal_delay_s',5),res.get('avg_hp_delay_s',5),res.get('energy_nj_bit',10000)
    norm_t=min(t/200.0,1.0);norm_p=p/100.0;norm_plr=plr/100.0;norm_nd=1.0-min(nd/5.0,1.0);norm_hd=1.0-min(hd/5.0,1.0);norm_e=1.0-min(e/10000.0,1.0)
    score=(0.3*norm_t+0.4*norm_p-0.1*norm_plr+0.1*norm_nd+0.2*norm_hd+0.1*norm_e)
    return np.clip(score,0.0,1.0)

def run_simulation(policy,steps,train_mode=False,run_id=1,agent=None,seed=BASE_SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);from ns3gym import ns3env
    cmd=f"./ns3 run '{SIM_SCRIPT} --openGymPort={PORT} --run={run_id}'";p=subprocess.Popen(cmd,cwd=NS3_PATH,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(3)
    env=None;original_epsilon=None
    try:
        env=ns3env.Ns3Env(port=PORT,startSim=False);s=np.array(env.reset(),dtype=np.float32)
        if agent is None:
            agent=PER_DoubleDQNAgent()
            if policy=="DQN" and not train_mode:
                try:agent.load(MODEL_PATH);agent.epsilon=0.0
                except FileNotFoundError:print("⚠️ Model not found. DQN is RANDOM.");policy="Random"
        elif policy=="DQN" and not train_mode:original_epsilon=agent.epsilon;agent.epsilon=0.0
        final_info={};bar=tqdm(range(steps),desc=f"{'Eval' if not train_mode else 'Train'} {policy} Run {run_id}")
        for step in bar:
            ps=s.copy()
            if train_mode and len(agent.memory)<agent.warmup:a=np.random.randint(ACTION_SIZE)
            elif policy=="DQN":a=agent.act(s)
            elif policy=="StrictPriority":
                if s[1]>0:a=0
                elif s[0]>25: a=5
                elif s[0]>0: a=1
                else: a=3
            else:a=np.random.randint(ACTION_SIZE)
            s_next,_,d,info_str=env.step(a);info=json.loads(info_str if info_str else'{}');info['action']=a;s_next=np.array(s_next,dtype=np.float32)
            if train_mode:
                r=calculate_reward(s_next,ps,info)
                agent.remember(s,a,r,s_next,d)
                agent.train()
            s=s_next
            if d:final_info=info;break
        return (final_info,agent)
    finally:
        if original_epsilon is not None and agent is not None:agent.epsilon=original_epsilon
        if env is not None:
            try:env.close()
            except:pass
        try:p.kill()
        except:pass
        subprocess.run(f"pkill -9 -f {SIM_SCRIPT}",shell=True,check=False);time.sleep(1)

if __name__=='__main__':
    print("🛠️ V53: Rebuilding NS-3...");build=subprocess.run(f"./ns3 build",cwd=NS3_PATH,shell=True,capture_output=True,text=True)
    if build.returncode!=0:print(f"❌ Build Fail:\n{build.stderr}");sys.exit(1)
    print("✅ Build OK.")
    training_agent=PER_DoubleDQNAgent();best_eval_score=float("-inf");best_epoch=0;train_log=[]
    print("\n🧠 V53: Training Final DQN...")
    for epoch in range(TRAINING_EPOCHS):
        epoch_seed=BASE_SEED+epoch;print(f"\n--- Epoch {epoch+1}/{TRAINING_EPOCHS} (Seed: {epoch_seed}) ---")
        _,training_agent=run_simulation("DQN",TRAIN_STEPS_PER_EPOCH,True,epoch+1,agent=training_agent,seed=epoch_seed)
        val_res,_=run_simulation("DQN",EVAL_STEPS,False,100+epoch,agent=training_agent)
        if not val_res:raise RuntimeError(f"Validation for epoch {epoch+1} failed to produce metrics.")
        score=get_eval_score(val_res);print(f"Epoch {epoch+1} Val Score: {score:.4f}");train_log.append({"epoch":epoch+1,"score":score})
        if score>best_eval_score:best_eval_score=score;best_epoch=epoch;training_agent.save(MODEL_PATH);print(f"  🥇 New best model saved (Score: {best_eval_score:.4f})")
        if epoch-best_epoch>=10:print("--- Early stopping triggered ---");break
    pd.DataFrame(train_log).to_csv("training_log_v53.csv",index=False)
    print("\n📊 V53: Final Evaluation...")
    results={"Proposed DQN-Edge":[],"Strict Priority":[],"Random":[]}
    policy_map={"Proposed DQN-Edge":"DQN","Strict Priority":"StrictPriority","Random":"Random"}
    for i in range(10):
        run_seed=BASE_SEED+200+i;print(f"\n--- Eval Run {i+1}/10 (Seed: {run_seed}) ---")
        for label,policy in policy_map.items():
            metrics,_=run_simulation(policy,EVAL_STEPS,False,i+1,seed=run_seed)
            if metrics:results[label].append(metrics)
            else:print(f"Warning: Empty metrics for {label} on run {i+1}")
    final={};
    for p,r in results.items():
        if not r:continue
        df=pd.DataFrame(r);final[p]={k:f"{df[k].mean():.4f}±{stats.t.ppf(0.975,len(df)-1)*df[k].std()/np.sqrt(len(df)):.4f}" for k in df.columns}
    summary_df=pd.DataFrame(final).T;print("\n\n--- FINAL RESULTS (V53, MEAN ± 95% CI) ---");print(summary_df.to_string())
    summary_df.to_excel("final_results_v53.xlsx");print("\n✅ Results exported to 'final_results_v53.xlsx'")
    if results["Proposed DQN-Edge"] and results["Strict Priority"]:
        for metric in results["Proposed DQN-Edge"][0].keys():
            dqn_data=[r[metric] for r in results["Proposed DQN-Edge"] if r and metric in r];sp_data=[r[metric] for r in results["Strict Priority"] if r and metric in r]
            n=min(len(dqn_data),len(sp_data))
            if n>1:t,p_val=stats.ttest_rel(dqn_data[:n],sp_data[:n]);print(f"T-test for {metric} (DQN vs SP): p-value={p_val:.4e}")
