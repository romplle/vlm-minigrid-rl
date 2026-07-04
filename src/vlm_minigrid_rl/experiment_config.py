from pathlib import Path

from .paths import project_path

BASE_MODEL_ID = "lusxvr/nanoVLM-222M"
WANDB_PROJECT = "nanoVLM-minigrid"

DEFAULT_ENV_SIZE = 8
DEFAULT_SEED = 42
DEFAULT_TILE_SIZE = 32
DEFAULT_NUM_DATASET_EPISODES = 1000
DEFAULT_EVAL_EPISODES = 250
DEFAULT_VAL_SAMPLES = 100

VAL_SPLIT_BY_ENV_SIZE = {8: 0.1, 16: 0.01}

DEFAULT_SFT_EPOCHS = 3
DEFAULT_GRPO_TRAIN_EPISODES = 100
DEFAULT_GRPO_CHECKPOINT_INTERVAL = 25
DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE = {8: 100, 16: 75}


def env_label(env_size: int) -> str:
    return f"{env_size}x{env_size}"


def dataset_dir(env_size: int) -> Path:
    return project_path(f"datasets/dataset_{env_label(env_size)}")


def sft_adapter_root(env_size: int) -> Path:
    return project_path(f"checkpoints/sft_adapter_{env_label(env_size)}")


def grpo_adapter_root(env_size: int) -> Path:
    return project_path(f"checkpoints/grpo_adapter_{env_label(env_size)}")


def sft_adapter_epoch_dir(env_size: int, epoch: int | None = None) -> Path:
    epoch_num = DEFAULT_SFT_EPOCHS if epoch is None else epoch
    return sft_adapter_root(env_size) / f"epoch-{epoch_num}"


def grpo_adapter_episode_dir(env_size: int, episode: int | None = None) -> Path:
    episode_num = (
        DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[env_size] if episode is None else episode
    )
    return grpo_adapter_root(env_size) / f"episode-{episode_num}"


def default_val_split(env_size: int) -> float:
    return VAL_SPLIT_BY_ENV_SIZE[env_size]


def default_grpo_eval_episode(env_size: int) -> int:
    return DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[env_size]


def adapter_config_exists(adapter_dir: Path) -> bool:
    return adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists()


def resolve_sft_adapter_path(env_size: int, epoch: int, root: Path | None = None) -> Path | None:
    adapter_dir = (root or sft_adapter_root(env_size)) / f"epoch-{epoch}"
    if adapter_config_exists(adapter_dir):
        return adapter_dir
    return None


def resolve_grpo_adapter_path(
    env_size: int,
    episode: int,
    root: Path | None = None,
) -> Path | None:
    adapter_root = root or grpo_adapter_root(env_size)
    episode_dir = adapter_root / f"episode-{episode}"
    if adapter_config_exists(episode_dir):
        return episode_dir
    if adapter_config_exists(adapter_root):
        return adapter_root
    return None
