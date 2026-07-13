import matplotlib.pyplot as plt
import re

def plot_training_rewards(log_file='running.txt'):
    """
    Parses the training log file and plots the average reward over episodes.
    """
    episodes = []
    avg_rewards = []

    # Regex to find lines with episode results
    # It captures the episode number and the average reward
    line_regex = re.compile(r"Ep\s+(\d+)\s+\|.*Avg Reward:\s+([\-0-9\.]+)")

    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = line_regex.search(line)
                if match:
                    episode = int(match.group(1))
                    avg_reward = float(match.group(2))
                    
                    episodes.append(episode)
                    avg_rewards.append(avg_reward)

        if not episodes:
            print("No reward data found in the log file. Was the format correct?")
            return

        # --- Plotting the results ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(episodes, avg_rewards, color='royalblue', linewidth=2, label='100-Episode Moving Average Reward')
        
        # Adding a title and labels
        ax.set_title('DQN Agent Learning Curve for Module 1', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        
        # Adding a trendline for better visualization of the learning progress
        # A simple moving average can smooth out the curve
        if len(avg_rewards) > 50:
            window_size = 50
            moving_avg = [np.mean(avg_rewards[i-window_size:i]) for i in range(window_size, len(avg_rewards))]
            ax.plot(episodes[window_size:], moving_avg, color='orangered', linestyle='--', linewidth=2, label=f'{window_size}-Episode Smoothed Trend')

        ax.legend(fontsize=10)
        ax.grid(True)
        
        print(f"Plot generated from {len(episodes)} episodes.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found. Make sure it's in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # We need numpy for the moving average calculation
    try:
        import numpy as np
        plot_training_rewards()
    except ImportError:
        print("Please install numpy: pip install numpy")

