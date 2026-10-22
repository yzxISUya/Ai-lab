"""项目的默认配置。

所有路径、模型结构、训练超参数和采样参数的默认值都集中在这里。
各脚本的命令行参数 Default 均引用本文件，显式传入的命令行参数会覆盖它们。

路径默认相对于运行命令时的当前目录（通常就是项目根目录）。
"""

from pathlib import Path

# 路径
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = Path("checkpoints")
RUN_DIR = CHECKPOINT_DIR / "run"

# 数据准备
DATASET_REPO_ID = "karpathy/tinystories-gpt4-clean"
DATASET_FILENAME = "tinystories_gpt4_clean.parquet"
MAX_CHARS = 200_000_000
VAL_FRACTION = 0.02
DATA_SEED = 42

# 模型
BLOCK_SIZE = 512
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.1

# 训练
BATCH_SIZE = 32
GRAD_ACCUM = 1
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 200
MAX_STEPS = 2_000
EVAL_STEPS = 200
EVAL_WINDOWS = 2_000
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 20
NUM_WORKERS = 2
TRAIN_SEED = 1337

# 采样
MODEL_DIR = RUN_DIR / "final"
PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 1024
NUM_SAMPLES = 3
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.1
SAMPLE_SEED = 42
