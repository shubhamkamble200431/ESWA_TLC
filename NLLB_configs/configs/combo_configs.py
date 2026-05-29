from configs.base_config import BaseConfig


def _make_combo(bbs: bool, ltc: bool, rope: bool, alb: bool) -> BaseConfig:
    cfg = BaseConfig()
    cfg.base_model        = "facebook/nllb-200-distilled-600M"
    cfg.num_train_epochs  = 20

    cfg.use_bucket_balanced_sampling = bbs

    cfg.use_length_token        = ltc
    cfg.use_length_penalty_loss = ltc

    cfg.use_rope   = rope
    cfg.use_alibi  = alb

    if rope:
        cfg.max_input_length = 1024

    flags = []
    if bbs:  flags.append("BBS")
    if ltc:  flags.append("LTC")
    if rope: flags.append("RoPE")
    if alb:  flags.append("ALiBi")
    cfg.description = (
        "Combinatorial ablation: baseline + "
        + (", ".join(flags) if flags else "nothing (≡ C-0)")
    )
    return cfg


def _x0000() -> BaseConfig:
    cfg = _make_combo(False, False, False, False)
    cfg.experiment_name = "X-0000_baseline"
    cfg.case_id   = "X-0000"
    cfg.output_dir = "outputs/X-0000"
    return cfg

def _x1000() -> BaseConfig:
    cfg = _make_combo(True, False, False, False)
    cfg.experiment_name = "X-1000_BBS"
    cfg.case_id   = "X-1000"
    cfg.output_dir = "outputs/X-1000"
    return cfg

def _x0100() -> BaseConfig:
    cfg = _make_combo(False, True, False, False)
    cfg.experiment_name = "X-0100_LTC"
    cfg.case_id   = "X-0100"
    cfg.output_dir = "outputs/X-0100"
    return cfg

def _x0010() -> BaseConfig:
    cfg = _make_combo(False, False, True, False)
    cfg.experiment_name = "X-0010_RoPE"
    cfg.case_id   = "X-0010"
    cfg.output_dir = "outputs/X-0010"
    return cfg

def _x0001() -> BaseConfig:
    cfg = _make_combo(False, False, False, True)
    cfg.experiment_name = "X-0001_ALiBi"
    cfg.case_id   = "X-0001"
    cfg.output_dir = "outputs/X-0001"
    return cfg

def _x1100() -> BaseConfig:
    cfg = _make_combo(True, True, False, False)
    cfg.experiment_name = "X-1100_BBS_LTC"
    cfg.case_id   = "X-1100"
    cfg.output_dir = "outputs/X-1100"
    return cfg

def _x1010() -> BaseConfig:
    cfg = _make_combo(True, False, True, False)
    cfg.experiment_name = "X-1010_BBS_RoPE"
    cfg.case_id   = "X-1010"
    cfg.output_dir = "outputs/X-1010"
    return cfg

def _x1001() -> BaseConfig:
    cfg = _make_combo(True, False, False, True)
    cfg.experiment_name = "X-1001_BBS_ALiBi"
    cfg.case_id   = "X-1001"
    cfg.output_dir = "outputs/X-1001"
    return cfg

def _x0110() -> BaseConfig:
    cfg = _make_combo(False, True, True, False)
    cfg.experiment_name = "X-0110_LTC_RoPE"
    cfg.case_id   = "X-0110"
    cfg.output_dir = "outputs/X-0110"
    return cfg

def _x0101() -> BaseConfig:
    cfg = _make_combo(False, True, False, True)
    cfg.experiment_name = "X-0101_LTC_ALiBi"
    cfg.case_id   = "X-0101"
    cfg.output_dir = "outputs/X-0101"
    return cfg

def _x0011() -> BaseConfig:
    cfg = _make_combo(False, False, True, True)
    cfg.experiment_name = "X-0011_RoPE_ALiBi"
    cfg.case_id   = "X-0011"
    cfg.output_dir = "outputs/X-0011"
    return cfg

def _x1110() -> BaseConfig:
    cfg = _make_combo(True, True, True, False)
    cfg.experiment_name = "X-1110_BBS_LTC_RoPE"
    cfg.case_id   = "X-1110"
    cfg.output_dir = "outputs/X-1110"
    return cfg

def _x1101() -> BaseConfig:
    cfg = _make_combo(True, True, False, True)
    cfg.experiment_name = "X-1101_BBS_LTC_ALiBi"
    cfg.case_id   = "X-1101"
    cfg.output_dir = "outputs/X-1101"
    return cfg

def _x1011() -> BaseConfig:
    cfg = _make_combo(True, False, True, True)
    cfg.experiment_name = "X-1011_BBS_RoPE_ALiBi"
    cfg.case_id   = "X-1011"
    cfg.output_dir = "outputs/X-1011"
    return cfg

def _x0111() -> BaseConfig:
    cfg = _make_combo(False, True, True, True)
    cfg.experiment_name = "X-0111_LTC_RoPE_ALiBi"
    cfg.case_id   = "X-0111"
    cfg.output_dir = "outputs/X-0111"
    return cfg

def _x1111() -> BaseConfig:
    cfg = _make_combo(True, True, True, True)
    cfg.experiment_name = "X-1111_BBS_LTC_RoPE_ALiBi"
    cfg.case_id   = "X-1111"
    cfg.output_dir = "outputs/X-1111"
    return cfg


ALL_COMBO_CONFIGS = {
    "X-0000": _x0000,
    "X-1000": _x1000,
    "X-0100": _x0100,
    "X-0010": _x0010,
    "X-0001": _x0001,
    "X-1100": _x1100,
    "X-1010": _x1010,
    "X-1001": _x1001,
    "X-0110": _x0110,
    "X-0101": _x0101,
    "X-0011": _x0011,
    "X-1110": _x1110,
    "X-1101": _x1101,
    "X-1011": _x1011,
    "X-0111": _x0111,
    "X-1111": _x1111,
}

COMBO_LABELS = {
    "X-0000": "Baseline (no extras)",
    "X-1000": "BBS",
    "X-0100": "LTC",
    "X-0010": "RoPE",
    "X-0001": "ALiBi",
    "X-1100": "BBS + LTC",
    "X-1010": "BBS + RoPE",
    "X-1001": "BBS + ALiBi",
    "X-0110": "LTC + RoPE",
    "X-0101": "LTC + ALiBi",
    "X-0011": "RoPE + ALiBi",
    "X-1110": "BBS + LTC + RoPE",
    "X-1101": "BBS + LTC + ALiBi",
    "X-1011": "BBS + RoPE + ALiBi",
    "X-0111": "LTC + RoPE + ALiBi",
    "X-1111": "BBS + LTC + RoPE + ALiBi",
}


def get_combo_config(case_id: str) -> BaseConfig:
    if case_id not in ALL_COMBO_CONFIGS:
        raise ValueError(
            f"Unknown combo case_id '{case_id}'. "
            f"Valid: {list(ALL_COMBO_CONFIGS)}"
        )
    return ALL_COMBO_CONFIGS[case_id]()
