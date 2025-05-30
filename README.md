# RLHF with DPO & PPO-RLHF

**Reinforcement Learning from Human Feedback: A Study of DPO and PPO-RLHF in Classic Control Environments**

This repository contains the code and experiments for our project comparing two preference-based reinforcement learning algorithms: **Direct Preference Optimization (DPO)** and **PPO-RLHF** on classic OpenAI Gym environments: `CartPole-v1` and `MountainCar-v0`.

---

## Overview

**Reinforcement Learning from Human Feedback (RLHF)** enables agents to learn behaviors aligned with human preferences, especially in environments where manually crafting a reward function is difficult or risky.

We evaluate two RLHF algorithms:

- **DPO (Direct Preference Optimization)**  
  Directly optimizes a policy using pairwise preferences, bypassing the need to model rewards.

- **PPO-RLHF (Proximal Policy Optimization with RLHF)**  
  Trains a reward model on preference data, then fine-tunes the policy using PPO with the learned reward model.

We generate synthetic preference data by comparing trajectories from expert and sub-expert policies and evaluate how well each method can recover near optimal behavior from this feedback.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/rosbotmay/rlhf_reward_hunters.git
cd rlhf_reward_hunters
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Reproducing Results

To reproduce the results from the paper:

- **CartPole-v1**  
  Please refer to the detailed instructions in [`cartpole/Readme.md`](cartpole/README.md)

- **MountainCar-v0**  
  Please refer to the detailed instructions in [`mountainCar/Readme.md`](mountainCar/README.md)

---

### Contact

For questions or issues, feel free to open a GitHub issue or reach out to any of the project contributors.
