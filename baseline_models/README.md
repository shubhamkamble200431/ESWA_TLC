# baseline_models

Baseline NMT systems for comparison against the NLLB combinatorial ablations.

| Model | Directory | HuggingFace ID |
|-------|-----------|----------------|
| mBART50 | `mBART50_many_to_many/` | `facebook/mbart-large-50-many-to-many-mmt` |
| mT5-Large | `mT5/` | `google/mt5-large` |
| IndicTrans2 | `indictrans2/` | AI4Bharat/indictrans2-indic-en-dist-200M |
| NLLB-200M | `nllb_200M_distilled/` | `facebook/nllb-200-distilled-600M` (no extras) |
| Mamba | `tf_mamba/` | Custom Mamba architecture |

Each directory (except tf_mamba) contains `train.py` and `inference_and_eval.py`.

## mBART50

```bash
python baseline_models/mBART50_many_to_many/train.py \
    --data_dir data --output_dir outputs/mBART50 --epochs 20 --fp16

python baseline_models/mBART50_many_to_many/inference_and_eval.py \
    --checkpoint outputs/mBART50/checkpoint-best \
    --data_dir data --split test
```

## mT5-Large

```bash
python baseline_models/mT5/train.py \
    --data_dir data --output_dir outputs/mT5 --epochs 20 --fp16

python baseline_models/mT5/inference_and_eval.py \
    --checkpoint outputs/mT5/checkpoint-best \
    --data_dir data --split test
```
