# data

| File | Description |
|------|-------------|
| `dataset.py` | `SANHINDataset`, bucket assignment, BBS sampler, concat augmentation, BT loading |

Key exports:
- `SANHINDataset` — PyTorch Dataset with optional length-token prepending
- `load_parallel_corpus(data_dir, src_file, tgt_file)` — load .sa/.hi files
- `build_weighted_sampler(dataset, bucket_weights)` — WeightedRandomSampler for BBS
- `concat_augment(src, tgt, ratio)` — synthetic B4-B5 pair generation
- `filter_by_buckets(src, tgt, allowed_buckets)` — curriculum filtering
