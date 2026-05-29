import os
import sys
import json
import argparse
import logging

import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID  = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG  = "hi_IN"
TGT_LANG  = "hi_IN"


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


class ParallelDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, tokenizer, max_len=256):
        assert len(src_lines) == len(tgt_lines)
        self.src = src_lines
        self.tgt = tgt_lines
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        self.tok.src_lang = SRC_LANG
        enc = self.tok(
            self.src[idx],
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        with self.tok.as_target_tokenizer():
            lab = self.tok(
                self.tgt[idx],
                max_length=self.max_len,
                truncation=True,
                padding=False,
            )
        label_ids = [t if t != self.tok.pad_token_id else -100 for t in lab["input_ids"]]
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         label_ids,
        }


def main():
    parser = argparse.ArgumentParser(description="mBART50 SAN→HIN fine-tuning")
    parser.add_argument("--data_dir",   default="data",    help="Dir with train.sa/hi, val.sa/hi")
    parser.add_argument("--output_dir", default="outputs/mBART50")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--max_len",    type=int,   default=256)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--fp16",       action="store_true")
    args = parser.parse_args()

    logger.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_ID, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    logger.info("Loading data...")
    train_src = load_lines(os.path.join(args.data_dir, "train.sa"))
    train_tgt = load_lines(os.path.join(args.data_dir, "train.hi"))
    val_src   = load_lines(os.path.join(args.data_dir, "val.sa"))
    val_tgt   = load_lines(os.path.join(args.data_dir, "val.hi"))
    logger.info(f"Train: {len(train_src):,}  Val: {len(val_src):,}")

    train_ds = ParallelDataset(train_src, train_tgt, tokenizer, args.max_len)
    val_ds   = ParallelDataset(val_src,   val_tgt,   tokenizer, args.max_len)

    logger.info(f"Loading model: {MODEL_ID}")
    model = MBartForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        logging_steps=50,
        save_total_limit=3,
        seed=args.seed,
        report_to=["none"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting training...")
    trainer.train()

    best_path = os.path.join(args.output_dir, "checkpoint-best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    logger.info(f"Model saved: {best_path}")


if __name__ == "__main__":
    main()
