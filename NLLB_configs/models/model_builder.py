import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
)

from configs.base_config import BaseConfig
from models.positional_encodings import apply_rope_to_model, apply_alibi_to_model
from models.hierarchical_encoder import HierarchicalEncoder

logger = logging.getLogger(__name__)

LENGTH_TOKENS = [f"<LEN_BUCKET_{i}>" for i in range(1, 6)]


def build_model_and_tokenizer(cfg: BaseConfig) -> Tuple[nn.Module, AutoTokenizer]:
    logger.info(f"[ModelBuilder] Loading tokenizer: {cfg.base_model}")
    tokenizer = _load_tokenizer(cfg)

    logger.info(f"[ModelBuilder] Loading model: {cfg.base_model}")
    model = _load_model(cfg, tokenizer)

    if cfg.use_length_token:
        _add_length_tokens(model, tokenizer)

    if cfg.extend_vocab_sanskrit:
        _extend_sanskrit_vocab(model, tokenizer)

    if cfg.use_lora:
        model = _apply_lora(model, cfg)

    if cfg.use_rope:
        _infer_head_dim_and_apply_rope(model, cfg)

    if cfg.use_alibi:
        _infer_nhead_and_apply_alibi(model, cfg)

    if cfg.use_hierarchical_encoder:
        model = _wrap_hierarchical_encoder(model, tokenizer, cfg)

    if cfg.use_flash_attention:
        _try_enable_flash_attention(model)

    _log_model_stats(model, cfg)
    return model, tokenizer


def _load_tokenizer(cfg: BaseConfig) -> AutoTokenizer:
    kwargs = {"use_fast": True}
    if "nllb" in cfg.base_model.lower():
        kwargs["src_lang"] = cfg.src_lang
        kwargs["tgt_lang"] = cfg.tgt_lang
    return AutoTokenizer.from_pretrained(cfg.base_model, **kwargs)


def _load_model(cfg: BaseConfig, tokenizer: AutoTokenizer) -> nn.Module:
    load_kwargs = {"dtype": torch.float32}
    if "mt5" in cfg.base_model.lower():
        return MT5ForConditionalGeneration.from_pretrained(cfg.base_model, **load_kwargs)
    elif "mbart" in cfg.base_model.lower():
        return MBartForConditionalGeneration.from_pretrained(cfg.base_model, **load_kwargs)
    else:
        return AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, **load_kwargs)


def _add_length_tokens(model, tokenizer):
    new_tokens = [t for t in LENGTH_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"[ModelBuilder] Added length tokens: {new_tokens}")


def _extend_sanskrit_vocab(model, tokenizer):
    devanagari_chars = [chr(c) for c in range(0x1C50, 0x1C80)]
    bigrams = [chr(a) + chr(b) for a in range(0x1C50, 0x1C70)
               for b in range(0x1C50, 0x1C70)]
    new_tokens = [t for t in devanagari_chars + bigrams[:200]
                  if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        old_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings()
            mean_emb = emb.weight[:old_size].mean(dim=0)
            emb.weight[old_size:] = mean_emb.unsqueeze(0).expand(
                emb.weight.shape[0] - old_size, -1)
        logger.info(f"[ModelBuilder] Extended vocab by {len(new_tokens)} OOV Sanskrit tokens.")


_LORA_MODULES_BY_FAMILY = {
    "m2m_100":   ["q_proj", "k_proj", "v_proj", "out_proj"],
    "mbart":     ["q_proj", "k_proj", "v_proj", "out_proj"],
    "mt5":       ["q", "k", "v", "o"],
    "t5":        ["q", "k", "v", "o"],
}


def _resolve_lora_target_modules(model, cfg) -> list:
    all_linear_names = set()
    for name, module in model.named_modules():
        if hasattr(module, "weight") and len(module.weight.shape) == 2:
            leaf = name.split(".")[-1]
            all_linear_names.add(leaf)

    def all_present(modules):
        return all(m in all_linear_names for m in modules)

    if all_present(cfg.lora_target_modules):
        return cfg.lora_target_modules

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    for family, modules in _LORA_MODULES_BY_FAMILY.items():
        if family in model_type.lower() and all_present(modules):
            logger.info(
                f"[ModelBuilder] cfg.lora_target_modules {cfg.lora_target_modules} "
                f"not found; using {family} defaults: {modules}"
            )
            return modules

    for family, modules in _LORA_MODULES_BY_FAMILY.items():
        if all_present(modules):
            logger.info(
                f"[ModelBuilder] Auto-detected LoRA target modules: {modules}"
            )
            return modules

    raise ValueError(
        f"Could not resolve LoRA target modules for model_type='{model_type}'. "
        f"Available linear leaf names: {sorted(all_linear_names)}. "
        f"Set cfg.lora_target_modules explicitly."
    )


def _apply_lora(model, cfg):
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError("Install peft: pip install peft")

    target_modules = _resolve_lora_target_modules(model, cfg)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    logger.info(
        f"[ModelBuilder] LoRA applied (rank={cfg.lora_rank}, "
        f"target_modules={target_modules})."
    )
    return model


def _infer_head_dim_and_apply_rope(model, cfg):
    mcfg = getattr(model, "config", None)
    head_dim = (mcfg.d_model // mcfg.encoder_attention_heads
                if mcfg and hasattr(mcfg, "d_model") else 64)
    apply_rope_to_model(model, head_dim=head_dim, max_seq_len=cfg.max_input_length)


def _infer_nhead_and_apply_alibi(model, cfg):
    mcfg = getattr(model, "config", None)
    num_heads = getattr(mcfg, "encoder_attention_heads", 16)
    apply_alibi_to_model(model, num_heads=num_heads, max_seq_len=cfg.max_input_length)


def _get_hf_model(model: nn.Module) -> nn.Module:
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.base_model.model
    except ImportError:
        pass
    return model


def _get_encoder(model: nn.Module) -> nn.Module:
    hf = _get_hf_model(model)

    if hasattr(hf, "model") and hasattr(hf.model, "encoder"):
        return hf.model.encoder

    if hasattr(hf, "encoder"):
        return hf.encoder

    raise AttributeError(
        f"Cannot find encoder in '{type(hf).__name__}'. "
        "Expected '.model.encoder' (NLLB/mBART) or '.encoder' (mT5)."
    )


def _set_encoder(model: nn.Module, new_encoder: nn.Module):
    hf = _get_hf_model(model)

    if hasattr(hf, "model") and hasattr(hf.model, "encoder"):
        hf.model.encoder = new_encoder
    elif hasattr(hf, "encoder"):
        hf.encoder = new_encoder
    else:
        raise AttributeError(
            f"Cannot set encoder on '{type(hf).__name__}'."
        )


def _get_model_config(model: nn.Module):
    if hasattr(model, "config"):
        return model.config
    return _get_hf_model(model).config


def _wrap_hierarchical_encoder(model, tokenizer, cfg):
    mcfg      = _get_model_config(model)
    d_model   = mcfg.d_model
    num_heads = getattr(mcfg, "encoder_attention_heads", 8)

    base_encoder = _get_encoder(model)

    hier_encoder = HierarchicalEncoder(
        base_encoder=base_encoder,
        d_model=d_model,
        tokenizer=tokenizer,
        segment_size=cfg.segment_size,
        cross_segment_layers=cfg.cross_segment_layers,
        nhead=num_heads,
        long_context_threshold=cfg.long_context_threshold,
    )

    _set_encoder(model, hier_encoder)
    logger.info(
        f"[ModelBuilder] HierarchicalEncoder injected "
        f"(segment_size={cfg.segment_size}, "
        f"cross_segment_layers={cfg.cross_segment_layers}, "
        f"threshold={cfg.long_context_threshold})."
    )
    return model


def _try_enable_flash_attention(model):
    try:
        model = model.to_bettertransformer()
        logger.info("[ModelBuilder] BetterTransformer (SDPA) enabled.")
    except Exception:
        logger.info(
            "[ModelBuilder] BetterTransformer not available — "
            "using standard attention. Install `optimum` for Flash Attention."
        )


def _log_model_stats(model, cfg):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[ModelBuilder] {cfg.experiment_name} | "
        f"Total params: {total/1e6:.1f}M | "
        f"Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)"
    )
