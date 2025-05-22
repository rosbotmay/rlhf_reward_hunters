import matplotlib.pyplot as plt

# Data
K_values = [500, 600, 700, 800, 900, 1000]
mean_returns = [286.00, 328.35, 354.68, 416.82, 431.63, 457.66]
std_devs = [15.23, 3.04, 10.94, 15.09, 10.27, 8.52]

# Create the figure
plt.figure(figsize=(10, 6))


error_bars = plt.errorbar(
    K_values, mean_returns, yerr=std_devs,
    fmt='none', ecolor='red', capsize=6, elinewidth=2, label='Std Deviation'
)

# Plot the main line
main_line = plt.plot(K_values, mean_returns, 'o-', linewidth=2.5, markersize=8, label='Policy trained by DPO')

# Add red error bars and store the bar container to label it


# Labels and title
plt.xlabel('Preference Data Size', fontsize=14)
plt.ylabel("Average Reward", fontsize=14)
plt.title('Mean Reward vs Preference Data Size', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Only horizontal grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.axhline(y=495, color='dimgray', linestyle='-.', linewidth=2, label='Optimal Policy $\pi_1$ (Avg Reward = 495)')
plt.axhline(y=350, color='gray', linestyle='--', linewidth=2, label='Fixed Policy $\pi_2$ (Avg Reward = 350)')
plt.text(K_values[-1] + 10, 350 + 1, r'$\pi_2$', fontsize=14, va='bottom', ha='left', color='gray')
plt.text(K_values[-1] + 10, 495 + 1, r'$\pi_1$', fontsize=14, va='bottom', ha='left', color='dimgray')

# Legend and layout
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('mean_reward_vs_preference_data_size.png', dpi=300, bbox_inches='tight')
plt.show()
