import pickle

import gym

from dpo import DPO, DPOModel
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from ppo_rhlf import PPO_RHLF, PPOPolicy, RewardModel



def plot_preferences_balance(preference_pairs):
    # Extract the preference values (prob_pi1) from preference_pairs
    preferences = [pair[2] for pair in preference_pairs]  # Get the preference values (prob_pi1)

    # Check if preferences list is empty
    if len(preferences) == 0:
        print("Error: No preference pairs available.")
        return

    # Count the number of times each policy is preferred
    preferred_pi1 = len([prob for prob in preferences if prob >= 0.5])
    preferred_pi2 = len(preferences) - preferred_pi1

    # Check if there are valid counts
    if preferred_pi1 == 0 and preferred_pi2 == 0:
        print("Error: No valid preference values (0 or 1) found in the data.")
        return

    # Prepare data for plotting
    labels = ['pi1 Preferred', 'pi2 Preferred']
    counts = [preferred_pi1, preferred_pi2]
    colors = ['skyblue', 'lightgreen']

    # Plotting the preference balance as a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Preference Balance')
    # Save the plot to a file
    plt.savefig("results/preference_balance.png")
    plt.close()




if __name__ == "__main__":
    # Load preference pairs from the pickle file
    with open("data/preference_pairs_200.pkl", "rb") as f:
        preference_pairs = pickle.load(f)
    
    # Inspect the loaded preference pairs
    print(f"Loaded {len(preference_pairs)} preference pairs.")
    plot_preferences_balance(preference_pairs)

    # Initialize DPO model and train it
    """dpo_model = DPOModel(input_size=6, output_size=1)  # 3 features from `get_features`
    dpo = DPO(dpo_model, preference_pairs,lr=1e-4)
    dpo.train(epochs=150)"""

    #not working for now..
    # Initialize PPO-RHLF and train it
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    """ppo_policy = PPOPolicy(state_size, action_size)
    reward_model = RewardModel(input_size=state_size + 1)  # Assuming state + action for the reward model

    ppo_rhlf = PPO_RHLF(ppo_policy, reward_model, env)
    ppo_rhlf.train(preference_pairs)"""

    ppo_policy = PPOPolicy(state_size, action_size)
    reward_model = RewardModel(input_size=state_size + 1) # Assuming no reward model is passed for simplicity

    # Train the PPO-RHLF agent
    ppo_rhlf = PPO_RHLF(ppo_policy, reward_model, env, learning_rate=1e-3, epochs=20)
    ppo_rhlf.train(preference_pairs)