"""
pretrained_dataset.py — Dataset and DataLoader for pretrained HuggingFace models.

Each model family uses its own tokeniser idiom:
  M2M100  : M2M100Tokenizer — needs forced_bos_token_id at decode time
  LongT5  : AutoTokenizer   — text2text prefix: "translate Sanskrit to Hindi: "
  mT5     : AutoTokenizer   — text2text prefix: "translate Sanskrit to Hindi: "
"""

import os
import logging
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, NUM_WORKERS, PretrainedConfig

logger = logging.getLogger(__name__)

# Prefix used by T5-family models
T5_PREFIX = "translate Sanskrit to Hindi: "


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


# ─────────────────────────────────────────────────────────────────
# Tokeniser factory
# ─────────────────────────────────────────────────────────────────

def load_hf_tokenizer(cfg: PretrainedConfig):
    """Load the appropriate HuggingFace tokeniser for a pretrained config."""
    from transformers import AutoTokenizer, M2M100Tokenizer

    if cfg.model_type == "m2m100":
        tok = M2M100Tokenizer.from_pretrained(cfg.hf_model_id)
        # Sanskrit ('sa') is not in M2M100's 100 languages.
        # We use 'hi' as the source-language token (closest supported
        # Devanagari language) while passing the raw Sanskrit text unchanged.
        # The model still translates correctly — the lang token only
        # guides the encoder's language-specific embeddings.
        m2m_src = cfg.src_lang if cfg.src_lang in tok.lang_code_to_token else "hi"
        m2m_tgt = cfg.tgt_lang if cfg.tgt_lang in tok.lang_code_to_token else "hi"
        if m2m_src != cfg.src_lang:
            logger.warning(
                "M2M100: src_lang '%s' not supported; falling back to '%s'",
                cfg.src_lang, m2m_src,
            )
        tok.src_lang = m2m_src
        tok.tgt_lang = m2m_tgt
    else:
        # mt5 and longt5 both use AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.hf_model_id)

    logger.info("Loaded tokeniser: %s  (type=%s)", cfg.hf_model_id, cfg.model_type)
    return tok


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class PretrainedTranslationDataset(Dataset):
    """
    Reads parallel .sa / .hi files.
    Returns raw string pairs so the collator can tokenise them on-the-fly
    (allows dynamic padding within each batch).
    """

    def __init__(self, src_file: str, tgt_file: str, model_type: str):
        self.model_type = model_type
        src_lines = _read_lines(src_file)
        tgt_lines = _read_lines(tgt_file)
        assert len(src_lines) == len(tgt_lines), (
            f"Mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt"
        )
        self.src_lines  = src_lines
        self.tgt_lines  = tgt_lines
        # raw word counts for bin-wise evaluation
        self.src_lengths = [len(s.split()) for s in src_lines]
        logger.info("Loaded %d pairs from %s / %s", len(src_lines), src_file, tgt_file)

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> Dict:
        src = self.src_lines[idx]
        if self.model_type in ("longt5", "mt5"):
            src = T5_PREFIX + src      # prepend task prefix for T5-family
        return {
            "src":        src,
            "tgt":        self.tgt_lines[idx],
            "src_length": self.src_lengths[idx],
        }


# ─────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────

class PretrainedCollator:
    """
    Tokenises a batch and returns HuggingFace-style input dicts.
    Handles M2M100 (forced_bos_token) vs T5-family (decoder_start_token).
    """

    def __init__(self, tokenizer, cfg: PretrainedConfig):
        self.tokenizer   = tokenizer
        self.cfg         = cfg
        self.max_src_len = cfg.max_src_len
        self.max_tgt_len = cfg.max_tgt_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        src_texts   = [b["src"]        for b in batch]
        tgt_texts   = [b["tgt"]        for b in batch]
        src_lengths = [b["src_length"] for b in batch]

        if self.cfg.model_type == "m2m100":
            self.tokenizer.src_lang = self.cfg.src_lang
            model_inputs = self.tokenizer(
                src_texts,
                max_length=self.max_src_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt_texts,
                    max_length=self.max_tgt_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
        else:
            # mt5 / longt5 — text2text style
            model_inputs = self.tokenizer(
                src_texts,
                max_length=self.max_src_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            labels = self.tokenizer(
                tgt_texts,
                max_length=self.max_tgt_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        # Replace pad token id with -100 so CrossEntropyLoss ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"]      = labels
        model_inputs["src_lengths"] = torch.tensor(src_lengths, dtype=torch.long)
        return model_inputs


# ─────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────

def get_pretrained_dataloaders(
    cfg: PretrainedConfig,
    tokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    collator = PretrainedCollator(tokenizer, cfg)

    def _loader(split: str, shuffle: bool) -> DataLoader:
        src_f = os.path.join(DATA_DIR, f"{split}.sa")
        tgt_f = os.path.join(DATA_DIR, f"{split}.hi")
        ds    = PretrainedTranslationDataset(src_f, tgt_f, cfg.model_type)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            collate_fn=collator,
            pin_memory=True,
        )

    return _loader("train", True), _loader("val", False), _loader("test", False)


def get_pretrained_test_dataset(cfg: PretrainedConfig) -> PretrainedTranslationDataset:
    src_f = os.path.join(DATA_DIR, "test.sa")
    tgt_f = os.path.join(DATA_DIR, "test.hi")
    return PretrainedTranslationDataset(src_f, tgt_f, cfg.model_type)