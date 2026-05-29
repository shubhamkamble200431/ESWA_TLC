"""
config.py — Central configuration for all experiments.

Models:
  Vanilla Transformer : 6, 12, 18 layers
  Mamba               : small, base, large
  Pretrained (HF)     : m2m100_418m, m2m100_1.2b, longt5_base, longt5_large,
                        mt5_small, mt5_base, mt5_large
"""

import os
from dataclasses import dataclass
from typing import List

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "65k_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
PRED_DIR   = os.path.join(OUTPUT_DIR, "predictions")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

for _d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR, PRED_DIR, PLOT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
# BPE / Tokeniser  (scratch models only)
# ─────────────────────────────────────────────
VOCAB_SIZE   = 8000
BPE_MODEL    = os.path.join(OUTPUT_DIR, "bpe.model")
BOS_TOKEN    = "<s>"
EOS_TOKEN    = "</s>"
PAD_TOKEN    = "<pad>"
UNK_TOKEN    = "<unk>"

# Language codes used by pretrained models
SRC_LANG = "sa"   # Sanskrit
TGT_LANG = "hi"   # Hindi

# ─────────────────────────────────────────────
# Training defaults
# ─────────────────────────────────────────────
MAX_LEN      = 128
BATCH_SIZE   = 64
NUM_WORKERS  = 4
SEED         = 42
MAX_EPOCHS   = 30
PATIENCE     = 15
WARMUP_STEPS = 4000
LABEL_SMOOTH = 0.1
CLIP_GRAD    = 1.0
DEVICE       = "cuda"

# ─────────────────────────────────────────────
# Bin boundaries (based on source sentence length)
# ─────────────────────────────────────────────
BINS: List[tuple] = [
    (1,  5),
    (6,  10),
    (11, 15),
    (16, 20),
    (21, None),
]
BIN_NAMES = ["1-5", "6-10", "11-15", "16-20", "20+"]


# ═════════════════════════════════════════════
# Scratch model configs
# ═════════════════════════════════════════════

@dataclass
class TransformerConfig:
    name: str
    model_type: str = "transformer"
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = MAX_LEN
    lr: float = 1e-3
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


@dataclass
class MambaConfig:
    name: str
    model_type: str = "mamba"
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 6
    dropout: float = 0.1
    max_len: int = MAX_LEN
    lr: float = 1e-3
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


# ═════════════════════════════════════════════
# Pretrained (HuggingFace) model config
# ═════════════════════════════════════════════

@dataclass
class PretrainedConfig:
    name: str
    model_type: str          # "m2m100" | "longt5" | "mt5"
    hf_model_id: str         # HuggingFace hub ID
    src_lang: str = SRC_LANG
    tgt_lang: str = TGT_LANG
    max_src_len: int = MAX_LEN
    max_tgt_len: int = MAX_LEN
    num_beams: int = 4
    lr: float = 5e-5
    batch_size: int = 16
    max_epochs: int = 20
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    fp16: bool = True        # set False if no GPU / no AMP support


# ── Transformer variants ─────────────────────
TRANSFORMER_CONFIGS = {
    "transformer_6":  TransformerConfig(name="transformer_6",  num_encoder_layers=6,  num_decoder_layers=6),
    "transformer_12": TransformerConfig(name="transformer_12", num_encoder_layers=12, num_decoder_layers=12),
    "transformer_18": TransformerConfig(name="transformer_18", num_encoder_layers=18, num_decoder_layers=18),
}

# ── Mamba variants ───────────────────────────
MAMBA_CONFIGS = {
    "mamba_small": MambaConfig(name="mamba_small", d_model=256, d_state=16, num_layers=6),
    "mamba_base":  MambaConfig(name="mamba_base",  d_model=512, d_state=32, num_layers=12),
    "mamba_large": MambaConfig(name="mamba_large", d_model=512, d_state=64, num_layers=18),
}

# ── Pretrained variants ──────────────────────
PRETRAINED_CONFIGS = {
    "m2m100_418m": PretrainedConfig(
        name="m2m100_418m",
        model_type="m2m100",
        hf_model_id="facebook/m2m100_418M",
        batch_size=16,
    ),
    "m2m100_1.2b": PretrainedConfig(
        name="m2m100_1.2b",
        model_type="m2m100",
        hf_model_id="facebook/m2m100_1.2B",
        batch_size=8,
    ),
    "longt5_base": PretrainedConfig(
        name="longt5_base",
        model_type="longt5",
        hf_model_id="google/long-t5-tglobal-base",
        batch_size=32,
        max_src_len=512,
    ),
    "longt5_large": PretrainedConfig(
        name="longt5_large",
        model_type="longt5",
        hf_model_id="google/long-t5-tglobal-large",
        batch_size=8,
        max_src_len=512,
    ),
    "mt5_small": PretrainedConfig(
        name="mt5_small",
        model_type="mt5",
        hf_model_id="google/mt5-small",
        batch_size=32,
    ),
    "mt5_base": PretrainedConfig(
        name="mt5_base",
        model_type="mt5",
        hf_model_id="google/mt5-base",
        batch_size=16,
    ),
    "mt5_large": PretrainedConfig(
        name="mt5_large",
        model_type="mt5",
        hf_model_id="google/mt5-large",
        batch_size=8,
    ),
}

# ── Combined lookups ─────────────────────────
ALL_SCRATCH_CONFIGS    = {**TRANSFORMER_CONFIGS, **MAMBA_CONFIGS}
ALL_PRETRAINED_CONFIGS = PRETRAINED_CONFIGS
ALL_CONFIGS            = {**ALL_SCRATCH_CONFIGS, **ALL_PRETRAINED_CONFIGS}