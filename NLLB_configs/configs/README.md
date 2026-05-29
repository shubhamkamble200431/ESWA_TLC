# configs

| File | Description |
|------|-------------|
| `base_config.py` | `BaseConfig` dataclass — all hyperparameters with defaults |
| `combo_configs.py` | 16 combinatorial X-* configs built from `BaseConfig` |

`combo_configs.py` exports:
- `get_combo_config(case_id)` — returns `BaseConfig` for given X-* ID
- `ALL_COMBO_CONFIGS` — dict of all 16 config callables
- `COMBO_LABELS` — human-readable labels per config
