
CHECKPOINTS_DIR = "checkpoints/"
RESULTS_DIR = "results/"
MODELS_DIR = "models/"
DATA_DIR = "data/"

ENV_NAME = "MountainCar-v0"

#### TRAJECTORY GENERATION CONFIGURATION ####
PI1_PATH = CHECKPOINTS_DIR + "ppo_checkpoint_1450_avg_reward_pi1"
PI2_PATH = CHECKPOINTS_DIR +  "ppo_checkpoint_250_avg_reward_pi2"
CHECKPOINT_PAIRS = [(PI1_PATH, PI2_PATH)]


# Number of preference pairs to generate per checkpoint pair
DATASIZES = [50, 100, 250, 500, 1000, 1500, 2000]
DATASIZES2 = [500, 600, 700, 1000]  # For reward model training
DATASIZES3 = [100,500,700,900,1000]  # For DPO training
SEEDS = [1024,256,32]









