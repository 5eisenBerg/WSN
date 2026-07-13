import matplotlib.pyplot as plt
import re
import numpy as np

def plot_training_rewards(log_file='running.txt', output_file='learning_curve.png'):
    """
    Parses the training log file, plots the average reward over episodes,
    and saves the plot to a file.
    """
    episodes = []
    avg_rewards = []

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
            print("No reward data found in the log file.")
            return

        # --- Plotting the results ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(episodes, avg_rewards, color='royalblue', linewidth=2, label='100-Episode Moving Average Reward')
        
        ax.set_title('DQN Agent Learning Curve for Module 1', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        
        # Adding a smoothed trendline
        if len(avg_rewards) > 50:
            window_size = 50
            # Calculate moving average, ensuring the arrays are the same length for plotting
            moving_avg = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(episodes[window_size-1:], moving_avg, color='orangered', linestyle='--', linewidth=2, label=f'{window_size}-Episode Smoothed Trend')

        ax.legend(fontsize=10)
        ax.grid(True)
        
        # --- FIX IS HERE: Save the figure instead of showing it ---
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"Plot successfully generated from {len(episodes)} episodes and saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    plot_training_rewards()
