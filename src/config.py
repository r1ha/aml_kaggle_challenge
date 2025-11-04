# src/config.py
import torch

# ===== Paths =====
DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/train/train.npz"
TEST_PATH = f"{DATA_DIR}/test/test.clean.npz"
SUBMISSION_PATH = "submission.csv"
CHECKPOINT_DIR = "checkpoints"

# ===== Model =====
INPUT_DIM = 1024
HIDDEN_DIM = 2048
OUTPUT_DIM = 1536
DROPOUT = 0.2

# ===== Training =====
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4

# ===== Device =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Evaluation =====
EVAL_BATCHES = 5
N_EVAL_SAMPLES = 500

# ===== Inference =====
INFER_BATCH_SIZE = 256
TOP_K = 10

# ===== Reproducibility =====
SEED = 42
