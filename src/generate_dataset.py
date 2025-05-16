import torch
from REINFORCE import *
import matplotlib.pyplot as plt
from utils import collect_bucket
import pickle
import os
from utils import make_pref_dataset, validate_prefs
from config import DATA_DIR

if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v1')
    env.reset(seed=0)

    # Print the observation and action space
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    # Generate expert's policy, pi1
    env = gym.make('CartPole-v1')
    env.reset(seed = 0)

    policy_pi1 = Policy().to(DEVICE)

    optimizer_pi1 = optim.Adam(policy_pi1.parameters(), lr=1e-3)
    scores_pi1 = reinforce_rwd2go_baseline(env, policy_pi1, optimizer_pi1, early_stop=True, n_episodes=2000, max_reward = 495)

    # Generate subexpert's policy, pi2
    env = gym.make('CartPole-v1')
    env.reset(seed=0)

    policy_pi2 = Policy().to(DEVICE)

    optimizer_pi2 = optim.Adam(policy_pi2.parameters(), lr=1e-3)
    scores_pi2 = reinforce_rwd2go_baseline(env, policy_pi2, optimizer_pi2, early_stop=True, n_episodes=2000, max_reward = 350)

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)

    # Plot the scores of pi1 and pi2
    # Using color-blind friendly hex codes
    ax.plot(np.arange(1, len(scores_pi1) + 1), scores_pi1, color='#0072B2', label='pi1')
    ax.plot(np.arange(1, len(scores_pi2) + 1), scores_pi2, color='#E69F00', label='pi2')

    ax.set_ylabel('Total reward (= time balanced)', fontsize=20)
    ax.set_xlabel('Episode #', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=20)
    
    fig_path = os.path.join(DATA_DIR, "training_curves_expert_vs_subexpert.png")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Saved training plot to {fig_path}")

    plt.show()

    # Save the policies
    torch.save(policy_pi1.state_dict(), os.path.join(DATA_DIR, 'policy_pi1.pth'))
    torch.save(policy_pi2.state_dict(), os.path.join(DATA_DIR, 'policy_pi2.pth'))


    env_name = "CartPole-v1"
    K = 1000 # number of trajectories to collect

    # expert π₁
    expert_trajs = collect_bucket(policy_pi1, env_name, K)

    #sub‐expert π₂
    subexpert_trajs = collect_bucket(policy_pi2, env_name, K)

    # uncomment if you want to collect random trajectorie
    # random_policy = RandomPolicy(gym.make(env_name).action_space)
    # random_trajs   = collect_bucket(random_policy, env_name, K)

    all_trajs = {
        "expert":    expert_trajs,
        "subexpert": subexpert_trajs,
        # "random":    random_trajs,
    }

    
    with open(os.path.join(DATA_DIR, 'trajectories.pkl'), "wb") as f:
        pickle.dump(all_trajs, f)

    # load the trajectories
    path = os.path.join(DATA_DIR, 'trajectories.pkl')

    with open(path, "rb") as f:
        all_trajs = pickle.load(f)

    # build each dataset (uncomment if you want pairs with random policy's trajectories)
    seed = 42
    ds_exp_sub   = make_pref_dataset(all_trajs["expert"],    all_trajs["subexpert"], seed)
    # ds_exp_rand  = make_pref_dataset(all_trajs["expert"],    all_trajs["random"],    seed)
    # ds_sub_rand  = make_pref_dataset(all_trajs["subexpert"], all_trajs["random"],    seed)

    prefs_path = os.path.join(DATA_DIR, "prefs_expert_vs_subexpert.pkl")

    with open(prefs_path, "wb") as f:
        pickle.dump(ds_exp_sub, f)

    # filename = "./data/prefs_expert_vs_random.pkl"
    # with open(filename, "wb") as f:
    #     pickle.dump(ds_exp_rand, f)

    # filename = "./data/prefs_subexpert_vs_random.pkl"
    # with open(filename, "wb") as f:
    #     pickle.dump(ds_sub_rand, f)

    print(f"expert vs subexpert: {len(ds_exp_sub)} pairs")
    # print(f"expert vs random   : {len(ds_exp_rand)} pairs")
    # print(f"subexpert vs random: {len(ds_sub_rand)} pairs")

    # Validate the preference dataset
    validate_prefs(prefs_path)