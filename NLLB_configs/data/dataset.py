import os
import random
import logging
from typing import List, Optional, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

BUCKET_BOUNDARIES = [5, 10, 15, 20]

def assign_bucket(n_tokens: int, boundaries: List[int] = BUCKET_BOUNDARIES) -> int:
    for i, b in enumerate(boundaries):
        if n_tokens <= b:
            return i + 1
    return len(boundaries) + 1


class SANHINDataset(Dataset):

    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        tokenizer: PreTrainedTokenizer,
        src_lang: str,
        tgt_lang: str,
        max_input_length: int = 512,
        max_target_length: int = 512,
        use_length_token: bool = False,
        bucket_boundaries: List[int] = None,
    ):
        assert len(src_sentences) == len(tgt_sentences), (
            f"Source ({len(src_sentences)}) and target ({len(tgt_sentences)}) "
            "must have equal line counts."
        )
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_length_token = use_length_token
        self.boundaries = bucket_boundaries or BUCKET_BOUNDARIES

        self.buckets: List[int] = [
            assign_bucket(len(s.split()), self.boundaries)
            for s in src_sentences
        ]

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        bucket = self.buckets[idx]

        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            src,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        tgt_text = f"<LEN_BUCKET_{bucket}> {tgt}" if self.use_length_token else tgt

        labels = self.tokenizer(
            text_target=tgt_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        label_ids = labels["input_ids"]
        pad_id = self.tokenizer.pad_token_id
        label_ids = [l if l != pad_id else -100 for l in label_ids]

        return {
            "input_ids":      model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels":         label_ids,
            "bucket":         bucket,
        }


def load_parallel_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parallel file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    logger.info(f"Loaded {len(lines):,} lines from {path}")
    return lines


def load_parallel_corpus(
    data_dir: str,
    src_file: str,
    tgt_file: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    src_path = os.path.join(data_dir, src_file)
    tgt_path = os.path.join(data_dir, tgt_file)
    src = load_parallel_file(src_path)
    tgt = load_parallel_file(tgt_path)
    assert len(src) == len(tgt), (
        f"Mismatched line counts: src={len(src)}, tgt={len(tgt)}"
    )
    if max_samples:
        src, tgt = src[:max_samples], tgt[:max_samples]
    return src, tgt


def concat_augment(
    src_sentences: List[str],
    tgt_sentences: List[str],
    ratio: float = 0.2,
    boundaries: List[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    boundaries = boundaries or BUCKET_BOUNDARIES
    short_indices = [
        i for i, s in enumerate(src_sentences)
        if assign_bucket(len(s.split()), boundaries) <= 2
    ]

    n_augment = int(len(src_sentences) * ratio)
    aug_src, aug_tgt = [], []
    conj_tokens = ["और", "लेकिन", "क्योंकि", "तथा"]

    for _ in range(n_augment):
        i, j = rng.sample(short_indices, 2)
        conj = rng.choice(conj_tokens)
        aug_src.append(f"{src_sentences[i]} {src_sentences[j]}")
        aug_tgt.append(f"{tgt_sentences[i]} {conj} {tgt_sentences[j]}")

    logger.info(
        f"Concat augmentation: added {len(aug_src):,} synthetic B4-B5 pairs."
    )
    return src_sentences + aug_src, tgt_sentences + aug_tgt


def load_bt_data(
    bt_dir: str,
    src_file: str = "bt.sa",
    tgt_file: str = "bt.hi",
) -> Tuple[List[str], List[str]]:
    if not os.path.isdir(bt_dir):
        logger.warning(f"BT data dir not found: {bt_dir} — skipping BT augmentation.")
        return [], []
    return load_parallel_corpus(bt_dir, src_file, tgt_file)


def filter_by_buckets(
    src: List[str],
    tgt: List[str],
    allowed_buckets: List[int],
    boundaries: List[int] = None,
) -> Tuple[List[str], List[str]]:
    boundaries = boundaries or BUCKET_BOUNDARIES
    filtered_src, filtered_tgt = [], []
    for s, t in zip(src, tgt):
        b = assign_bucket(len(s.split()), boundaries)
        if b in allowed_buckets:
            filtered_src.append(s)
            filtered_tgt.append(t)
    logger.info(
        f"Curriculum filter buckets={allowed_buckets}: "
        f"{len(filtered_src):,} / {len(src):,} pairs retained."
    )
    return filtered_src, filtered_tgt


def build_weighted_sampler(
    dataset: SANHINDataset,
    bucket_weights: List[float],
) -> WeightedRandomSampler:
    weights = [bucket_weights[b - 1] for b in dataset.buckets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def build_dataloader(
    dataset: SANHINDataset,
    batch_size: int,
    use_bucket_balanced_sampling: bool = False,
    bucket_weights: Optional[List[float]] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    if use_bucket_balanced_sampling and bucket_weights:
        sampler = build_weighted_sampler(dataset, bucket_weights)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
