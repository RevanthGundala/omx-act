from dataclasses import dataclass


@dataclass(frozen=True)
class ACTConfig:
    # Dataset
    dataset_repo_id: str
    dataset_revision: str = "main"
    dataset_format: str = "lerobot"
    camera_names: tuple[str, ...] = ("wrist", "top")
    benchmark_dataset_dir: str | None = None
    benchmark_task_name: str | None = None
    benchmark_num_episodes: int | None = None
    benchmark_episode_len: int | None = None

    # Run identity
    job_name: str = "act_pour_water"
    output_root: str = "outputs/act_experiments"

    # Data loading
    batch_size: int = 8
    chunk_size: int = 100
    num_workers: int = 4
    train_split: float = 0.95

    # Model
    d_model: int = 512
    joint_dim: int = 6
    d_z: int = 32
    num_vae_encoder_layers: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1 # known bug in main act implementation
    num_heads: int = 8
    mlp_dim: int = 3200
    dropout: float = 0.1
    exp_weight: float = 0.01

    # Optimization
    num_train_steps: int = 100_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000  # NOTE: not currently wired into a scheduler in models/act/train.py;
    # LR stays constant at `learning_rate` throughout training, matching LeRobot's own ACT policy
    # (get_scheduler_preset() -> None, no scheduler either). Kept as a documented no-op rather
    # than silently removed, in case a scheduler is added later.
    grad_clip: float = 10.0  # matches LeRobot's AdamWConfig default (lerobot/optim/optimizers.py,
    # grad_clip_norm=10.0), confirmed via the actual official checkpoint's train_config.json.
    # Previously 1.0 here -- 10x stricter than what trained the official 20k checkpoint we
    # compare against; our custom run's own logged grad_norm was ~5-6 for most of training
    # (see outputs/vastai/instance-43177076/.../metrics.jsonl), meaning the old value of 1.0
    # was actively clamping gradients on 100% of logged steps, while LeRobot's 10.0 would only
    # have engaged on ~55% of them.
    kl_weight: float = 10.0
    dropout: float = 0.1
    seed: int = 1000

    # Logging/checkpointing
    log_freq: int = 100
    eval_freq: int = 1000
    save_freq: int = 5000