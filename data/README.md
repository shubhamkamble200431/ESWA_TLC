# data

Parallel Sanskrit–Hindi corpus.

| File | Lines | Description |
|------|-------|-------------|
| `train.sa` / `train.hi` | 58,106 | Training set |
| `val.sa` / `val.hi` | ~7,000 | Validation set |
| `test.sa` / `test.hi` | 7,264 | Raw test set |
| `test_7264_5.sa` / `test_7264_5.hi` | 7,260 | Test set resampled to 1452 per bucket |

File format: plain text, one sentence per line, UTF-8 Devanagari.

Generate `test_7264_5.*` from the raw test files:

```bash
python utils/resample_7264_5.py \
    --src data/test.sa --tgt data/test.hi \
    --out_src data/test_7264_5.sa --out_tgt data/test_7264_5.hi
```
