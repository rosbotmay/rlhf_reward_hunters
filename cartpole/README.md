# rlhf_reward_hunters
EE-568 Reinforcement learning project

### How to generate preference data

Run it with `python .\generate_data.py`

This script performs the following steps in sequence:

1. **Setup & Environment Initialization**  
   - Set random seeds for reproducibility (`torch.manual_seed(0)`).  
   - Select device (CPU/GPU).  
   - Create and seed the Gym `CartPole-v1` environment.  
   - Print the observation and action spaces.

2. **Train “Expert” Policy π₁**  
   - Re-initialize environment and instantiate `Policy()` → `policy_pi1`.  
   - Optimize via `reinforce_rwd2go_baseline` (REINFORCE with learned baseline), stopping once average reward ≥ 495 or 2 000 episodes.  
   - Collect episode returns in `scores_pi1`.

3. **Train “Sub-expert” Policy π₂**  
   - Re-initialize environment and instantiate a fresh `Policy()` → `policy_pi2`.  
   - Optimize via basic `reinforce`, stopping once average reward ≥ 250 or 2 000 episodes.  
   - Collect episode returns in `scores_pi2`.

4. **Plot Learning Curves**  
   - Plot `scores_pi1` (blue) vs. `scores_pi2` (orange) over episodes using Matplotlib.  
   - Label axes, add legend, and display the figure.

5. **Save Trained Policies**  
   - Export `policy_pi1.pth` and `policy_pi2.pth` into `data/` via `torch.save`.

6. **Collect Trajectories**  
   - Call `collect_bucket(policy_pi1, env_name, K)` and `collect_bucket(policy_pi2, env_name, K)` for K rollouts each.  
   - Store in a dict:
     ```python
     {
       "expert": expert_trajs,
       "subexpert": subexpert_trajs,
     }
     ```
   - Pickle to `data/trajectories.pkl`.

7. **Build Preference Dataset**  
   - Reload `trajectories.pkl`.  
   - Generate labeled pairs with:
     ```python
     ds_exp_sub = make_pref_dataset(expert_trajs, subexpert_trajs, seed=42)
     ```
   - Pickle to `data/prefs_expert_vs_subexpert.pkl`.

8. **Validate Preference Dataset**  
   - Run `validate_prefs("data/prefs_expert_vs_subexpert.pkl")` to check:
     - Label balance (fraction of expert vs. sub-expert wins)  
     - Return‐difference statistics  
     - Bradley–Terry probability distribution

---

**Notes:**  
- You can uncomment the random‐policy lines to also generate `expert vs. random` and `sub-expert vs. random` datasets.  
- Adjust `K`, learning rates, and stopping thresholds as needed for your experiments.  



### How to Run DPO

Run it with `python .\train_dpo.py`

This script fine-tunes a pretrained policy using Direct Preference Optimization (DPO) and then evaluates its performance.


1. **Instantiate the Policy**  
   - Creates a new `Policy` object matching the CartPole actor network.  
   - Moves it to the configured compute device (CPU or GPU).

2. **Load Expert Weights**  
   - Reads the checkpoint `policy_pi1.pth` from your `data/` folder (we want to fine-tune pi1 with our preference pairs).  
   - Loads those weights into the `policy` object so it now behaves like your “expert” π₁.

3. **Run DPO Fine-Tuning**  
   - Calls `train_dpo(...)`, passing in:
     - The loaded expert policy as the starting point.  
     - The filename of your preference dataset (`prefs_expert_vs_subexpert.pkl`).  
   - Inside `train_dpo`, the policy is updated over multiple epochs to better match the trajectory preferences, and the resulting fine-tuned model is saved as `dpo_policy_expert_sub.pth` in your `models/` folder.

4. **Evaluate the Fine-Tuned Policy**  
   - Constructs the path to the newly saved DPO checkpoint.  
   - Calls `evaluate_dpo_policy(...)`, which:
     - Loads that checkpoint into a fresh `Policy` instance.  
     - Runs 50 rollouts in the real CartPole environment.  
     - Records each episode’s return and computes the mean and standard deviation.

5. **Report Results**  
   - Prints out the average (`mean return`) and variability (`std`) of those 50 evaluation episodes so you can see whether DPO actually improved the true performance.


### How to Run PPO-RLHF

#### 1. Train the PPO Subexpert
```bash
python train_ppo_subexpert.py
```

This trains a PPO policy to suboptimal performance and saves it as: `checkpoints/ppo_subexpert.zip`

#### 2. Train Reward Model + Fine-Tune PPO
```bash
python sb3_ppo.py
```

- Trains a reward model using preference data  
- Fine-tunes the subexpert with PPO using the learned rewards  
- Saves models like: `checkpoints/ppo_rlhf_cartpole_500`

#### 3. Evaluate PPO-RLHF vs Subexpert

```bash
python sb3_evaluation.py
```
- Loads and evaluates both the PPO subexpert and the PPO-RLHF models  
- Prints:
  - Mean rewards  
  - 95% confidence intervals  
  - Absolute and relative improvement

  ---

**In essence**, these scripts take the pretrained expert policy, refine it using either DPO or PPO-RLHF on the saved preference pairs, and then evaluate how well the fine-tuned policy performs under the true environment reward—i.e., how well it balances the pole in CartPole.
