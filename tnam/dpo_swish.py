import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from tqdm import tqdm
import wandb
import pickle

from utils import load_agent
from dqn_pendulum import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        # Differential initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)  # Critical for stable init

    def forward(self, x):
        x = F.silu(self.fc1(x))  # Swish activation
        x = F.silu(self.fc2(x))
        return self.fc3(x)

def reference_log_policy(agent, state, tau=1.0):
    state = agent.state_normalizer.normalize(state)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent.q_network(state)
    logits = q_values / tau
    log_probs = F.log_softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return log_probs

def train_dpo(preference_dataset, pi_2, num_epochs=20, beta=0.3, 
             policy_temp=0.5, ref_temp=0.5, grad_penalty=0.1):
    policy = PolicyNetwork(3, 50).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, 
                                weight_decay=1e-4)
    
    # Store initial reference values as buffers
    with torch.no_grad():
        initial_ref_pref = np.mean([pi_2.state_normalizer.normalize(s[0]) 
                                  for traj in preference_dataset 
                                  for s in traj[0]])
        initial_ref_rej = np.mean([pi_2.state_normalizer.normalize(s[0]) 
                               for traj in preference_dataset 
                               for s in traj[1]])
    
    for epoch in range(num_epochs):
        policy.train()
        total_metrics = {'loss': 0, 'policy_grad': 0, 
                        'advantage': 0, 'entropy': 0}
        
        for preferred, rejected in preference_dataset:
            # Batch processing with trajectory slicing
            states_pref = torch.stack([torch.FloatTensor(s[0]) 
                                     for s in preferred[:50]]).to(device)
            actions_pref = torch.LongTensor([s[1] 
                                           for s in preferred[:50]]).to(device)
            
            # Policy outputs with learned temperature
            logits_pref = policy(states_pref) / policy_temp
            log_probs_pref = F.log_softmax(logits_pref, dim=1)
            log_p_pref = log_probs_pref.gather(1, actions_pref.unsqueeze(1)).mean()
            
            # Repeat for rejected trajectories
            states_rej = torch.stack([torch.FloatTensor(s[0]) 
                                    for s in rejected[:50]]).to(device)
            actions_rej = torch.LongTensor([s[1] 
                                          for s in rejected[:50]]).to(device)
            logits_rej = policy(states_rej) / policy_temp
            log_p_rej = F.log_softmax(logits_rej, dim=1).gather(1, actions_rej.unsqueeze(1)).mean()
            
            # Reference policy with detached computation
            with torch.no_grad():
                ref_logp_pref = np.mean([reference_log_policy(pi_2, s[0], tau=ref_temp)[a]
                                      for s, a in zip(preferred[:50], actions_pref.cpu().numpy())])
                ref_logp_rej = np.mean([reference_log_policy(pi_2, s[0], tau=ref_temp)[a]
                                     for s, a in zip(rejected[:50], actions_rej.cpu().numpy())])
            
            # Dynamic advantage calculation
            advantage = (log_p_pref - ref_logp_pref) - (log_p_rej - ref_logp_rej)
            
            # Stabilized loss with gradient penalty
            loss = -F.logsigmoid(beta * advantage) + grad_penalty * (log_p_pref**2 + log_p_rej**2)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient monitoring and clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            
            optimizer.step()
            
            # Metrics collection
            total_metrics['loss'] += loss.item()
            total_metrics['policy_grad'] += total_grad_norm.item()
            total_metrics['advantage'] += advantage.item()
            total_metrics['entropy'] += (-log_probs_pref.exp() * log_probs_pref).sum(1).mean().item()
        
        # Epoch logging
        avg_metrics = {k: v/len(preference_dataset) for k, v in total_metrics.items()}
        wandb.log({
            **avg_metrics,
            "reference_logp/preferred": ref_logp_pref,
            "reference_logp/rejected": ref_logp_rej,
            "policy_logp/preferred": log_p_pref.item(),
            "policy_logp/rejected": log_p_rej.item(),
            "temp/policy": policy_temp,
            "temp/reference": ref_temp
        }, step=epoch)
        
        print(f"Epoch {epoch+1}: "
              f"Loss: {avg_metrics['loss']:.3f} | "
              f"Grad: {avg_metrics['policy_grad']:.2f} | "
              f"Entropy: {avg_metrics['entropy']:.2f}")
    
    return policy

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Initialize WandB
    wandb.init(
        project="rlhf-pendulum",
        name="dpo-training-swish",
        config={
            "algorithm": "DPO",
            "environment": "Pendulum-v1"
        }
    )
    
    # Define action space
    action_space = np.linspace(-2.0, 2.0, 50)
    
    # Load pi_2 from checkpoint
    checkpoint_path = "/home/thai/rlhf/checkpoints_dqn_5000/dqn_checkpoint_episode_best_at_4903.pt"
    pi_2 = load_agent(checkpoint_path, state_dim=3, action_space=action_space)
    
    # Load preference dataset
    with open('preference_dataset.pkl', 'rb') as f:
        preference_dataset = pickle.load(f)
    
    # Train DPO
    train_dpo(
        preference_dataset,
        pi_2,
        num_epochs=20,
        beta=0.3,  # Reduced from 0.5 to prevent over-regularization
        policy_temp=0.3,  # Lower temperature for sharper policy
        ref_temp=0.7,  # Higher temp for softer reference
        grad_penalty=0.05  # Reduced regularization
    )
    
    # Save final model
    torch.save(policy.state_dict(), "dpo_policy_new_514.pt")
    wandb.save("dpo_policy_new_514.pt")
    
    # Finish WandB run
    wandb.finish()