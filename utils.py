import numpy as np

def evaluate_ppo_model(model,env,episodes=30,render=False):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        done=False
        cum_rewards = 0
        while not done:
            # Use the PPO model to predict the next action
            action, _ = model.predict(state, deterministic=True)  # Use deterministic action selection (exploitation)
            # Step the environment
            next_state, reward, done,info = env.step(action)
            cum_rewards += reward
            # Update the state
            state = next_state
            if render:
                env.render()
        rewards.append(cum_rewards)
    print(f"Model average over {episodes} episodes : {np.mean(rewards)} with std : {np.std(rewards)}")
    return rewards