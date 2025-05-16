import os, torch

ENV_NAME       = "CartPole-v1"
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')

# # PPO hyperparams
# TOTAL_STEPS     = 200_000
# BATCH_SIZE      = 2048
# EPOCHS_PER_BATCH= 10
# GAMMA, LAMBDA   = 0.99, 0.95
# CLIP_EPS        = 0.2
# VF_COEF, ENT_COEF = 0.5, 0.01
# LR_PI, LR_VF    = 3e-4, 1e-3
# EVAL_INTERVAL   = BATCH_SIZE
# N_EVAL_EPISODES = 5

# DPO hyperparams
DPO_LR       = 1e-3
DPO_EPOCHS   = 10
BATCH_SIZE_DPO = 64
DPO_LAMBDA  = 0.1
DPO_CLIP_EPS = 0.2
