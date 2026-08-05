import pandas as pd
import numpy as np

# --- V75 HP Verification Diagnostic ---
LOG_FILE = 'deep_diagnostic_log.csv'
ACTION_LABELS = ['Send HP', 'Send Normal', 'Drop Normal', 'Sleep', 'Drop HP']

def verify_hp_traffic():
    print(f"--- 🔬 V75: Verifying HP Traffic from '{LOG_FILE}' ---")
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: '{LOG_FILE}' not found. Please run the V71 deep diagnostic script first.")
        return

    # Question 1: Are HP packets being generated?
    hp_present_steps = df[df['s_q_hp'] > 0]
    num_hp_present = len(hp_present_steps)
    if num_hp_present > 0:
        print(f"\n✅ SUCCESS: HP packets were present in the queue for {num_hp_present} out of {len(df)} steps ({num_hp_present/len(df)*100:.1f}% of the time).")
    else:
        print("\n❌ CRITICAL FAILURE: HP packets were never generated or never reached the queue.")
        return

    # Question 2: Are HP packets being transmitted?
    hp_send_actions = df[df['action'] == 0]
    num_hp_sent = len(hp_send_actions)
    if num_hp_sent > 0:
        print(f"✅ SUCCESS: The agent chose 'Send HP' (Action 0) {num_hp_sent} times.")
        # We can't directly see 'hp_sent' from this log, but choosing the action is strong evidence.
    else:
        print("\n❌ CRITICAL FAILURE: The agent NEVER chose to send an HP packet, confirming the policy collapse.")
        # This is what we expect from the V68 model
    
    # Question 3: Is the delay truly zero?
    # We analyze the state WHEN HP packets are present.
    avg_hp_age_when_present = hp_present_steps['s_age_hp'].mean()
    print(f"\n🔬 Analysis of HP Packet Age:")
    print(f"  - Average 'age_hp' of the oldest HP packet when the HP queue was not empty: {avg_hp_age_when_present:.6f}")

    # Now, let's check the age of HP packets at the moment the agent chose to send them (if it ever did).
    if num_hp_sent > 0:
        avg_age_at_send = hp_send_actions['s_age_hp'].mean()
        print(f"  - Average 'age_hp' at the moment the agent chose 'Send HP': {avg_age_at_send:.6f}")
        if avg_age_at_send < 0.01: # Assuming simulation step is 100ms (0.1s)
             print("\n✅ CONCLUSION: The HP delay is effectively zero. The agent is transmitting HP packets almost immediately after they arrive.")
        else:
             print("\n⚠️ WARNING: The average delay is NOT zero. There is a discrepancy between the final metrics and the diagnostic log.")
    else:
        # This is the V68 case.
        print("\n❌ CONCLUSION (for V68 model): The agent ignores aging HP packets, leading to high eventual delay or timeouts, which is not reflected in the `0.0000` final metric because no HP packets were ever successfully delivered by the DQN.")


if __name__ == "__main__":
    verify_hp_traffic()
