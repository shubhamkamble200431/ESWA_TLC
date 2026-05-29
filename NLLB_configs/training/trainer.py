import os
import inspect
import logging
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from configs.base_config import BaseConfig
from data.dataset import (
    SANHINDataset,
    build_weighted_sampler,
    filter_by_buckets,
)
from training.loss import LengthAwareLoss
from training.curriculum import CurriculumScheduler

logger = logging.getLogger(__name__)


class SANHINTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        cfg: BaseConfig,
        tokenizer,
        full_train_src: List[str],
        full_train_tgt: List[str],
        *args,
        **kwargs,
    ):
        self.cfg = cfg
        self.san_tokenizer = tokenizer
        self.full_train_src = full_train_src
        self.full_train_tgt = full_train_tgt
        self.curriculum = CurriculumScheduler() if cfg.use_curriculum else None
        self._current_epoch = 1

        self.custom_loss = (
            LengthAwareLoss(
                pad_token_id=tokenizer.pad_token_id,
                length_penalty_weight=cfg.length_penalty_weight,
            )
            if cfg.use_length_penalty_loss
            else None
        )

        super().__init__(*args, **kwargs)

        plot_dir = os.path.join(cfg.output_dir, "plots")
        self.add_callback(LivePlotCallback(
            save_dir=plot_dir,
            case_id=cfg.experiment_name,
        ))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("bucket", None)

        if self.custom_loss is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs.pop("bucket", None)
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )

    def get_train_dataloader(self) -> DataLoader:
        epoch = self._current_epoch

        if self.curriculum:
            self.curriculum.log_stage(epoch)
            active_buckets = self.curriculum.get_active_buckets(epoch)
            filtered_src, filtered_tgt = filter_by_buckets(
                self.full_train_src,
                self.full_train_tgt,
                allowed_buckets=active_buckets,
                boundaries=self.cfg.bucket_boundaries,
            )
            use_balanced = self.curriculum.use_balanced_sampling(epoch)
        else:
            filtered_src = self.full_train_src
            filtered_tgt = self.full_train_tgt
            use_balanced = self.cfg.use_bucket_balanced_sampling

        epoch_dataset = SANHINDataset(
            src_sentences=filtered_src,
            tgt_sentences=filtered_tgt,
            tokenizer=self.san_tokenizer,
            src_lang=self.cfg.src_lang,
            tgt_lang=self.cfg.tgt_lang,
            max_input_length=self.cfg.max_input_length,
            max_target_length=self.cfg.max_target_length,
            use_length_token=self.cfg.use_length_token,
            bucket_boundaries=self.cfg.bucket_boundaries,
        )

        _base_collator = DataCollatorForSeq2Seq(
            tokenizer=self.san_tokenizer,
            model=self.model,
            padding=True,
        )

        def collator(features):
            for f in features:
                f.pop("bucket", None)
            return _base_collator(features)

        if use_balanced:
            sampler = build_weighted_sampler(epoch_dataset, self.cfg.bucket_weights)
            return DataLoader(
                epoch_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=collator,
                num_workers=self.cfg.dataloader_num_workers,
                pin_memory=True,
            )

        return DataLoader(
            epoch_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.cfg.dataloader_num_workers,
            pin_memory=True,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        result = super().training_step(model, inputs, num_items_in_batch)
        if hasattr(self, "state") and self.state.global_step > 0:
            steps_per_epoch = max(
                1,
                len(self.full_train_src)
                // self.args.per_device_train_batch_size
                // self.args.gradient_accumulation_steps,
            )
            self._current_epoch = (self.state.global_step // steps_per_epoch) + 1
        return result


def _resolve_report_to():
    try:
        import tensorboard  # noqa: F401
        return ["tensorboard"]
    except ImportError:
        try:
            import tensorboardX  # noqa: F401
            return ["tensorboard"]
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "[Trainer] tensorboard not found — disabling report_to. "
                "Install with: pip install tensorboard"
            )
            return ["none"]

def build_training_args(cfg: BaseConfig) -> Seq2SeqTrainingArguments:
    try:
        import torch
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)

    strategy = getattr(cfg, "eval_strategy", "epoch")

    if "eval_strategy" in sig.parameters:
        eval_strategy_kwargs = {"eval_strategy": strategy}
    else:
        eval_strategy_kwargs = {"evaluation_strategy": strategy}

    extra_kwargs = {}
    if "sortish_sampler" in sig.parameters:
        extra_kwargs["sortish_sampler"] = False

    step_kwargs = {}
    if strategy == "steps":
        step_kwargs["eval_steps"]  = getattr(cfg, "eval_steps",  500) or 500
        step_kwargs["save_steps"]  = getattr(cfg, "save_steps",  500) or 500

    max_tgt = cfg.max_target_length
    if "generation_max_new_tokens" in sig.parameters:
        gen_len_kwargs = {"generation_max_new_tokens": max_tgt}
    else:
        gen_len_kwargs = {"generation_max_length": max_tgt}

    gen_beams = getattr(cfg, "train_eval_num_beams", 1)
    if "generation_num_beams" in sig.parameters:
        extra_kwargs["generation_num_beams"] = gen_beams

    if "torch_compile" in sig.parameters:
        extra_kwargs["torch_compile"] = False

    if "save_safetensors" in sig.parameters:
        extra_kwargs["save_safetensors"] = False

    return Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        deepspeed=None,
        ddp_find_unused_parameters=False,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        **eval_strategy_kwargs,
        **step_kwargs,
        save_strategy=strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_steps=cfg.logging_steps,
        predict_with_generate=True,
        **gen_len_kwargs,
        report_to=_resolve_report_to(),
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        run_name=cfg.experiment_name,
        **extra_kwargs,
    )


from transformers import TrainerCallback
import os, json
import numpy as np

class LivePlotCallback(TrainerCallback):

    def __init__(self, save_dir: str, case_id: str = "experiment"):
        self.save_dir = save_dir
        self.case_id  = case_id
        os.makedirs(save_dir, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            self._has_mpl = True
        except ImportError:
            self._has_mpl = False
            import logging
            logging.getLogger(__name__).warning(
                "matplotlib not found — skipping live plots. "
                "Install with: pip install matplotlib"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self._has_mpl:
            return
        try:
            self._plot(state)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[LivePlot] Plot failed: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if not self._has_mpl:
            return
        try:
            self._plot(state, final=True)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[LivePlot] Final plot failed: {e}")

    def _plot(self, state, final=False):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as mgridspec

        history = state.log_history

        train_steps, train_loss = [], []
        eval_steps, eval_bleu   = [], []
        bucket_series = {f"B{i}": ([], []) for i in range(1, 6)}
        lr_steps, lr_vals       = [], []

        bucket_tag_map = {
            "B1": "eval/bleu_B1_1-5",
            "B2": "eval/bleu_B2_6-10",
            "B3": "eval/bleu_B3_11-15",
            "B4": "eval/bleu_B4_16-20",
            "B5": "eval/bleu_B5_20+",
        }

        for entry in history:
            step = entry.get("step", 0)
            if "loss" in entry and "eval_loss" not in entry:
                train_steps.append(step)
                train_loss.append(entry["loss"])
            if "learning_rate" in entry:
                lr_steps.append(step)
                lr_vals.append(entry["learning_rate"])
            if "eval_bleu" in entry:
                eval_steps.append(step)
                eval_bleu.append(entry.get("eval_bleu", 0))
            for bk, tag in bucket_tag_map.items():
                clean = tag.replace("eval/", "eval_").replace("-", "_").replace("+", "plus")
                val = entry.get(clean) or entry.get(tag)
                if val is not None:
                    bucket_series[bk][0].append(step)
                    bucket_series[bk][1].append(val)

        DARK   = "#0a1628"
        PANEL  = "#0f172a"
        GRID   = "#1e293b"
        TEXT   = "#e2e8f0"
        MUTED  = "#64748b"
        COLORS = {
            "loss": "#60a5fa", "bleu": "#34d399",
            "lr":   "#facc15",
            "B1": "#60a5fa", "B2": "#34d399", "B3": "#a78bfa",
            "B4": "#fb923c", "B5": "#f472b6",
        }
        BLABELS = {
            "B1": "B1 (1–5)", "B2": "B2 (6–10)", "B3": "B3 (11–15)",
            "B4": "B4 (16–20)", "B5": "B5 (20+)",
        }

        def style_ax(ax, title, xlabel="Step", ylabel=""):
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
            ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)

        def smooth(vals, w=5):
            if len(vals) < w:
                return vals
            arr = np.array(vals, dtype=float)
            k = np.ones(w) / w
            s = np.convolve(arr, k, mode="same")
            for i in range(w//2):
                s[i] = arr[:i+1].mean()
                s[-(i+1)] = arr[-(i+1):].mean()
            return s.tolist()

        fig = plt.figure(figsize=(18, 10), facecolor=DARK)
        suffix = "FINAL" if final else f"step{state.global_step}"
        fig.suptitle(
            f"Training Dashboard — {self.case_id}  |  SAN→HIN NMT  [{suffix}]",
            color=TEXT, fontsize=13, fontweight="bold", y=0.99,
        )
        gs = mgridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1, "Training Loss", ylabel="Loss")
        if train_loss:
            ax1.plot(train_steps, smooth(train_loss), color=COLORS["loss"], lw=2)
            ax1.fill_between(train_steps, smooth(train_loss), alpha=0.12,
                             color=COLORS["loss"])

        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2, "Validation BLEU (Overall)", ylabel="BLEU")
        if eval_bleu:
            ax2.plot(eval_steps, eval_bleu, color=COLORS["bleu"],
                     lw=2, marker="o", markersize=4)
            ax2.fill_between(eval_steps, eval_bleu, alpha=0.12, color=COLORS["bleu"])

        ax4 = fig.add_subplot(gs[1, 0:2])
        style_ax(ax4, "Per-Bucket BLEU over Training", ylabel="BLEU")
        has_bucket = False
        for bk, (bsteps, bvals) in bucket_series.items():
            if bvals:
                ax4.plot(bsteps, bvals, color=COLORS[bk],
                         label=BLABELS[bk], lw=2, marker="o", markersize=3)
                has_bucket = True
        if has_bucket:
            ax4.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID,
                       labelcolor=MUTED, ncol=5, loc="upper left")
        else:
            ax4.text(0.5, 0.5, "Bucket metrics appear after first eval",
                     ha="center", va="center", transform=ax4.transAxes,
                     color=MUTED, fontsize=9)

        ax5 = fig.add_subplot(gs[1, 2])
        style_ax(ax5, "Learning Rate Schedule", ylabel="LR")
        if lr_vals:
            ax5.plot(lr_steps, lr_vals, color=COLORS["lr"], lw=2)
            ax5.set_yscale("log")

        path_dash = os.path.join(self.save_dir, f"{self.case_id}_dashboard.png")
        fig.savefig(path_dash, dpi=140, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)

        last_bucket_vals = []
        for bk in ["B1", "B2", "B3", "B4", "B5"]:
            bvals = bucket_series[bk][1]
            last_bucket_vals.append(bvals[-1] if bvals else 0)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
        fig2.suptitle(f"Bucket Analysis — {self.case_id}  [{suffix}]",
                      color=TEXT, fontsize=12, fontweight="bold")

        ax_bar = axes2[0]
        ax_bar.set_facecolor(PANEL)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor(GRID)
        ax_bar.grid(True, color=GRID, linewidth=0.7, axis="y", alpha=0.7)
        ax_bar.set_title("Per-Bucket BLEU (Latest)", color=TEXT, fontsize=10)
        ax_bar.set_ylabel("BLEU", color=MUTED, fontsize=9)
        ax_bar.tick_params(colors=MUTED, labelsize=8)
        blabel_list = [BLABELS[f"B{i}"] for i in range(1, 6)]
        bcolor_list  = [COLORS[f"B{i}"] for i in range(1, 6)]
        bars = ax_bar.bar(blabel_list, last_bucket_vals,
                          color=bcolor_list, edgecolor=DARK, linewidth=0.5)
        for bar, val in zip(bars, last_bucket_vals):
            if val > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.3,
                            f"{val:.1f}", ha="center", va="bottom",
                            fontsize=9, color=TEXT)
        ax_lr = axes2[1]
        ax_lr.set_facecolor(PANEL)
        for spine in ax_lr.spines.values():
            spine.set_edgecolor(GRID)
        ax_lr.grid(True, color=GRID, linewidth=0.7, axis="y", alpha=0.7)
        ax_lr.set_title("Length Ratio per Bucket (Latest)", color=TEXT, fontsize=10)
        ax_lr.set_ylabel("pred_len / ref_len", color=MUTED, fontsize=9)
        ax_lr.tick_params(colors=MUTED, labelsize=8)
        lr_tag_map = {
            "B1": "eval_length_ratio_B1_1_5",
            "B2": "eval_length_ratio_B2_6_10",
            "B3": "eval_length_ratio_B3_11_15",
            "B4": "eval_length_ratio_B4_16_20",
            "B5": "eval_length_ratio_B5_20plus",
        }
        lr_vals_bar = []
        for bk in ["B1", "B2", "B3", "B4", "B5"]:
            tag = lr_tag_map[bk]
            val = next(
                (e.get(tag, 0) for e in reversed(history) if tag in e), 0
            )
            lr_vals_bar.append(val)
        ax_lr.bar(blabel_list, lr_vals_bar, color=bcolor_list,
                  edgecolor=DARK, linewidth=0.5)
        ax_lr.axhline(1.0, color="#facc15", linestyle="--",
                      linewidth=1.3, label="Target = 1.0")
        ax_lr.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=MUTED)
        for i, val in enumerate(lr_vals_bar):
            if val > 0:
                ax_lr.text(i, val + 0.01, f"{val:.3f}",
                           ha="center", va="bottom", fontsize=8, color=TEXT)

        path_bucket = os.path.join(
            self.save_dir, f"{self.case_id}_bucket_analysis.png"
        )
        fig2.savefig(path_bucket, dpi=140, bbox_inches="tight", facecolor=DARK)
        plt.close(fig2)
