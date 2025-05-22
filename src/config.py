import os, torch

ENV_NAME       = "CartPole-v1"
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')

# # PPO hyperparams
BATCH_SIZE      = 512
PPO_EPOCHS      = 6         # number of PPO minibatch passes
MINI_BATCH      = 128       # minibatch size for PPO epochs
GAMMA           = 0.99      # discount
LAM             = 0.95      # GAE lambda
CLIP_EPS        = 0.2       # PPO clipping epsilon
LR_PI           = 3e-4      # policy LR
LR_V            = 1e-3      # value LR

# DPO hyperparams
DPO_LR       = 1e-3
DPO_EPOCHS   = 10
BATCH_SIZE_DPO = 64
DPO_LAMBDA  = 0.1
DPO_CLIP_EPS = 0.2
