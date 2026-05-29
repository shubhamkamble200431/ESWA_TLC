from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BaseConfig:
    experiment_name: str = "C-0_baseline"
    case_id: str = "C-0"
    description: str = "Baseline NLLB fine-tune, no modifications"

    data_dir: str = "data"
    train_src: str = "train.sa"
    train_tgt: str = "train.hi"
    val_src: str = "val.sa"
    val_tgt: str = "val.hi"
    test_src: str = "test_7264_5.sa"
    test_tgt: str = "test_7264_5.hi"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    base_model: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "san_Deva"
    tgt_lang: str = "hin_Deva"
    max_input_length: int = 512
    max_target_length: int = 128
    comet_eval_every: int = 5
    train_eval_num_beams: int = 1
    final_eval_num_beams: int = 4

    use_rope: bool = False
    use_alibi: bool = False
    position_interpolation_scale: float = 1.0

    use_hierarchical_encoder: bool = False
    segment_size: int = 12
    cross_segment_layers: int = 2
    long_context_threshold: int = 20

    extend_vocab_sanskrit: bool = False
    transliterate_to_devanagari: bool = False

    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    use_madx_adapters: bool = False
    use_prefix_tuning: bool = False
    prefix_length: int = 100

    use_length_token: bool = False
    use_length_penalty_loss: bool = False
    length_penalty_weight: float = 0.1

    use_bucket_balanced_sampling: bool = False
    bucket_boundaries: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20]
    )
    bucket_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.2, 1.5, 2.0]
    )

    use_curriculum: bool = False
    curriculum_stages: List[dict] = field(default_factory=lambda: [
        {"epochs": (1, 5),   "buckets": [1, 2, 3]},
        {"epochs": (6, 12),  "buckets": [1, 2, 3, 4, 5]},
        {"epochs": (13, 20), "buckets": [1, 2, 3, 4, 5], "balanced": True},
    ])

    use_back_translation: bool = False
    bt_data_path: Optional[str] = None
    use_concat_augmentation: bool = False
    concat_aug_ratio: float = 0.2
    use_noise_injection: bool = False

    use_flash_attention: bool = True
    use_longformer_attention: bool = False
    longformer_window_size: int = 64

    num_train_epochs: int = 20
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 2
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_bleu"
    greater_is_better: bool = True

    output_dir: str = "outputs"
    logging_steps: int = 50
    eval_steps: int = None
    save_steps: int = None
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    seed: int = 42
