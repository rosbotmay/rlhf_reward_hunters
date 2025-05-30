# RLHF Reward Hunters: MountainCar

This project implements Reinforcement Learning from Human Feedback (RLHF) methods for the classic MountainCar environment. It includes implementations of PPO, PPO-RLHF, DPO, and reward modeling, along with scripts for training and evaluation.

## Project Structure

```
config.py                  # Configuration settings
dpo.py                     # Direct Preference Optimization (DPO) training
ppo.py                     # Proximal Policy Optimization (PPO) training
ppo_rlhf.py                # PPO with RLHF training
reward.py                  # Reward model training and inference
trajectories.py            # Trajectory and data handling
utils.py                   # Utility functions
models/                    # Saved reward models and policies
checkpoints/               # PPO checkpoints
data/                      # Human preference data
results/                   # Output results and plots
```

## Setup

1. **Install dependencies**  
   Make sure you have Python 3.8+ and install required packages:
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare Data**  
   Place your human preference data in the `data/` directory. Example files:
   - `preference_pairs_500_original.pkl`
   - `preference_pairs_2000_original.pkl`
   or run trajectories.py

## Usage

### 1. Train a Reward Model

Required : data/preference_pairs_500_original.pkl
output : models/reward_model_500.pth

```sh
python reward.py 
```

### 2. Train a Policy with PPO

```sh
python ppo.py
```

### 3. Train with PPO-RLHF

```sh
python ppo_rlhf.py
```

### 4. Train with DPO

```sh
python dpo.py
```

## Results

- Trained models and checkpoints are saved in the `models/` and `checkpoints/` directories.
- Plots and evaluation metrics are saved in the `results/` directory.
