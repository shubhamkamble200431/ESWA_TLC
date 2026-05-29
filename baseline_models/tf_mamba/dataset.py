"""
dataset.py — BPE tokenisation + PyTorch Dataset / DataLoader.

Trains a shared SentencePiece BPE model on the source+target training
files, then wraps each split as a TranslationDataset.
"""

import os
import logging
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

from config import (
    DATA_DIR, OUTPUT_DIR, BPE_MODEL,
    VOCAB_SIZE, MAX_LEN, BOS_TOKEN, EOS_TOKEN,
    PAD_TOKEN, UNK_TOKEN, NUM_WORKERS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# BPE  tokeniser helpers
# ─────────────────────────────────────────────────────────────────

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def train_bpe(force: bool = False) -> None:
    """Train a shared BPE model on train.sa + train.hi."""
    if os.path.exists(BPE_MODEL) and not force:
        logger.info("BPE model already exists at %s — skipping training.", BPE_MODEL)
        return

    src_path = os.path.join(DATA_DIR, "train.sa")
    tgt_path = os.path.join(DATA_DIR, "train.hi")
    combined = os.path.join(OUTPUT_DIR, "_combined_train.txt")

    with open(combined, "w", encoding="utf-8") as fout:
        for p in [src_path, tgt_path]:
            for line in _read_lines(p):
                fout.write(line.strip() + "\n")

    prefix = BPE_MODEL.replace(".model", "")
    spm.SentencePieceTrainer.train(
        input=combined,
        model_prefix=prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        character_coverage=0.9999,
        num_threads=4,
    )
    logger.info("BPE model saved to %s.model", prefix)


def load_tokenizer() -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL)
    return sp


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Reads parallel .sa / .hi files and returns integer-encoded pairs.
    Source lengths (in raw tokens before BPE) are stored in self.src_lengths
    for bin-wise evaluation.
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        sp: spm.SentencePieceProcessor,
        max_len: int = MAX_LEN,
    ):
        self.sp      = sp
        self.max_len = max_len

        src_lines = _read_lines(src_file)
        tgt_lines = _read_lines(tgt_file)
        assert len(src_lines) == len(tgt_lines), (
            f"Mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt"
        )

        self.src_raw = src_lines
        self.tgt_raw = tgt_lines

        # raw word counts (space-split) for binning — use source file
        self.src_lengths = [len(s.split()) for s in src_lines]

        # encode
        self.src_ids: List[List[int]] = []
        self.tgt_ids: List[List[int]] = []

        bos = sp.bos_id()
        eos = sp.eos_id()

        for s, t in zip(src_lines, tgt_lines):
            src_enc = sp.encode(s, out_type=int)[:max_len]
            tgt_enc = sp.encode(t, out_type=int)[:max_len - 2]
            self.src_ids.append(src_enc)
            self.tgt_ids.append([bos] + tgt_enc + [eos])

        logger.info(
            "Loaded %d sentence pairs from %s / %s",
            len(self.src_ids), src_file, tgt_file,
        )

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.tensor(self.src_ids[idx], dtype=torch.long),
            torch.tensor(self.tgt_ids[idx], dtype=torch.long),
            self.src_lengths[idx],
        )


def collate_fn(batch):
    """Pad within a batch to the max length in that batch."""
    src_list, tgt_list, lengths = zip(*batch)
    pad_id = 0  # PAD token id
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded, torch.tensor(lengths, dtype=torch.long)


def get_dataloaders(
    sp: spm.SentencePieceProcessor,
    batch_size: int = 64,
    max_len: int = MAX_LEN,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train / val / test DataLoaders."""

    def _loader(split: str, shuffle: bool) -> DataLoader:
        src_f = os.path.join(DATA_DIR, f"{split}.sa")
        tgt_f = os.path.join(DATA_DIR, f"{split}.hi")
        ds = TranslationDataset(src_f, tgt_f, sp, max_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    train_loader = _loader("train", shuffle=True)
    val_loader   = _loader("val",   shuffle=False)
    test_loader  = _loader("test_7264_5",  shuffle=False)
    return train_loader, val_loader, test_loader


def get_test_dataset(sp: spm.SentencePieceProcessor, max_len: int = MAX_LEN) -> TranslationDataset:
    src_f = os.path.join(DATA_DIR, "test_7264_5.sa")
    tgt_f = os.path.join(DATA_DIR, "test_7264_5.hi")
    return TranslationDataset(src_f, tgt_f, sp, max_len)