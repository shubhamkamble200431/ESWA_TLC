# training

| File | Description |
|------|-------------|
| `trainer.py` | `SANHINTrainer` — extends `Seq2SeqTrainer` with curriculum and BBS |
| `loss.py` | `LengthAwareLoss` (NLL + length penalty) and `BucketBalancedLoss` |
| `curriculum.py` | `CurriculumScheduler` — 3-stage bucket curriculum |

Curriculum stages:
- Epochs 1–5: B1-B3 only
- Epochs 6–12: B1-B5
- Epochs 13–20: B1-B5 + BBS
