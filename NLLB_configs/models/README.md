# models

| File | Description |
|------|-------------|
| `model_builder.py` | Master builder: loads NLLB/mBART/mT5, applies LoRA, RoPE/ALiBi, HierarchicalEncoder |
| `positional_encodings.py` | RoPE (Su et al.) and ALiBi (Press et al.) implementations |
| `hierarchical_encoder.py` | Two-tier encoder for B5 sentences (20+ tokens) |

Main entry point: `build_model_and_tokenizer(cfg)` returns `(model, tokenizer)`.
