import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import gymnasium as gym

from tqdm import tqdm
import wandb
import pickle

from utils import load_agent
from dqn_pendulum import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

def reference_log_policy(agent, state):
    state = agent.state_normalizer.normalize(state)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent.q_network(state)
    tau = 1.0
    logits = q_values / tau
    log_probs = F.log_softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return log_probs

def train_dpo(preference_dataset, pi_2, num_epochs=50, lr=1e-3, beta=0.05, patience=5):
    # Split dataset into training and validation sets (80-20 split)
    train_dataset, val_dataset = train_test_split(preference_dataset, test_size=0.2, random_state=42)
    
    policy = PolicyNetwork(state_dim=3, action_dim=50).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    # Initialize cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Log hyperparameters
    wandb.config.update({
        "num_epochs": num_epochs,
        "beta": beta,
        "learning_rate": lr,
        "patience": patience,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "device": str(device)
    })
    
    for epoch in tqdm(range(num_epochs), desc="DPO Epochs"):
        policy.train()  # Set to training mode
        total_train_loss = 0
        
        # Training loop
        for preferred_traj, rejected_traj in train_dataset:
            # Process preferred trajectory
            states_pref = torch.FloatTensor([step[0] for step in preferred_traj]).to(device)
            actions_pref = torch.LongTensor([step[1] for step in preferred_traj]).to(device)
            logits_pref = policy(states_pref)
            log_probs_pref = F.log_softmax(logits_pref, dim=1)
            log_p_pref = log_probs_pref.gather(1, actions_pref.unsqueeze(1)).sum()
            
            # Process rejected trajectory
            states_rej = torch.FloatTensor([step[0] for step in rejected_traj]).to(device)
            actions_rej = torch.LongTensor([step[1] for step in rejected_traj]).to(device)
            logits_rej = policy(states_rej)
            log_probs_rej = F.log_softmax(logits_rej, dim=1)
            log_p_rej = log_probs_rej.gather(1, actions_rej.unsqueeze(1)).sum()
            
            # Calculate reference policy log probs
            log_p_ref_pref = sum(reference_log_policy(pi_2, step[0])[step[1]] for step in preferred_traj)
            log_p_ref_rej = sum(reference_log_policy(pi_2, step[0])[step[1]] for step in rejected_traj)
            
            # DPO loss calculation
            ratio = (log_p_pref - log_p_ref_pref) - (log_p_rej - log_p_ref_rej)
            loss = -torch.log(torch.sigmoid(beta * ratio))
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # Validation phase
        policy.eval()  # Set to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for preferred_traj, rejected_traj in val_dataset:
                states_pref = torch.FloatTensor([step[0] for step in preferred_traj]).to(device)
                actions_pref = torch.LongTensor([step[1] for step in preferred_traj]).to(device)
                logits_pref = policy(states_pref)
                log_probs_pref = F.log_softmax(logits_pref, dim=1)
                log_p_pref = log_probs_pref.gather(1, actions_pref.unsqueeze(1)).sum()
                
                states_rej = torch.FloatTensor([step[0] for step in rejected_traj]).to(device)
                actions_rej = torch.LongTensor([step[1] for step in rejected_traj]).to(device)
                logits_rej = policy(states_rej)
                log_probs_rej = F.log_softmax(logits_rej, dim=1)
                log_p_rej = log_probs_rej.gather(1, actions_rej.unsqueeze(1)).sum()
                
                log_p_ref_pref = sum(reference_log_policy(pi_2, step[0])[step[1]] for step in preferred_traj)
                log_p_ref_rej = sum(reference_log_policy(pi_2, step[0])[step[1]] for step in rejected_traj)
                
                ratio = (log_p_pref - log_p_ref_pref) - (log_p_rej - log_p_ref_rej)
                loss = -torch.log(torch.sigmoid(beta * ratio))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataset)
        
        # Log metrics to WandB
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch+1)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(policy.state_dict(), "dpo_policy_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Step the scheduler
        scheduler.step()
    
    # Load the best model
    policy.load_state_dict(torch.load("dpo_policy_best.pt"))
    return policy

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Initialize WandB
    wandb.init(
        project="rlhf-pendulum",
        name="dpo-training",
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
    policy = train_dpo(preference_dataset, pi_2, num_epochs=100, beta=0.099, lr=1e-4, patience=5)
    
    # Save final model
    torch.save(policy.state_dict(), "dpo_policy_best.pt")
    wandb.save("dpo_policy_best.pt")
    
    # Finish WandB run
    wandb.finish()