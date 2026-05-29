# scripts

| File | Description |
|------|-------------|
| `back_translate.py` | Generate synthetic SAN sentences via HIN→SAN back-translation |

Usage:
```bash
python scripts/back_translate.py \
    --hi_mono_file ../data/hindi_mono.txt \
    --out_dir ../data/bt_augmented \
    --confidence_threshold -2.5
```
