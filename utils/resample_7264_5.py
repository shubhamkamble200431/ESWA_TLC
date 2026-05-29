import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)


def wc(s):
    return len(s.split())


def get_bucket(sa):
    w = wc(sa)
    if   w <= 5:  return "b1"
    elif w <= 10: return "b2"
    elif w <= 15: return "b3"
    elif w <= 20: return "b4"
    else:         return "b5"


def resample(bucket_data, target, pool):
    n = len(bucket_data)
    if n >= target:
        return random.sample(bucket_data, target)
    needed = target - n
    pool_sample = []
    while len(pool_sample) < needed:
        take = min(needed - len(pool_sample), len(pool))
        pool_sample.extend(random.sample(pool, take))
    return list(bucket_data) + pool_sample[:needed]


def main():
    parser = argparse.ArgumentParser(description="Resample test set to equal bucket sizes")
    parser.add_argument("--src",       required=True, help="Input Sanskrit file (.sa)")
    parser.add_argument("--tgt",       required=True, help="Input Hindi file (.hi)")
    parser.add_argument("--out_src",   required=True, help="Output resampled Sanskrit file")
    parser.add_argument("--out_tgt",   required=True, help="Output resampled Hindi file")
    parser.add_argument("--plot",      default=None,  help="Optional path to save plot (.png)")
    args = parser.parse_args()

    sa_lines = open(args.src, encoding="utf-8").read().strip().split("\n")
    hi_lines = open(args.tgt, encoding="utf-8").read().strip().split("\n")
    assert len(sa_lines) == len(hi_lines)
    pairs = list(zip(sa_lines, hi_lines))

    TOTAL   = len(sa_lines)
    TARGET  = TOTAL // 5
    BUCKETS = ["b1", "b2", "b3", "b4", "b5"]

    buckets = {k: [] for k in BUCKETS}
    for sa, hi in pairs:
        buckets[get_bucket(sa)].append((sa, hi))

    print(f"Total lines  : {TOTAL}")
    print(f"Target/bucket: {TARGET}")
    for k in BUCKETS:
        status = "downsample" if len(buckets[k]) >= TARGET else "upsample"
        print(f"  {k}: {len(buckets[k]):>5}  → {status}")

    supplement_pool = (
        [(sa, hi) for sa, hi in buckets["b1"] if wc(sa) >= 3]
        + buckets["b2"]
    )
    random.shuffle(supplement_pool)
    print(f"Supplement pool (b1>=3w + b2): {len(supplement_pool)}")

    out = {k: resample(buckets[k], TARGET, supplement_pool) for k in BUCKETS}

    print("Resampled sizes:")
    for k in BUCKETS:
        print(f"  {k}: {len(out[k])}")

    all_pairs = [pair for k in BUCKETS for pair in out[k]]
    random.shuffle(all_pairs)
    assert len(all_pairs) == TARGET * 5

    with open(args.out_src, "w", encoding="utf-8") as f:
        f.write("\n".join(sa for sa, hi in all_pairs))
    with open(args.out_tgt, "w", encoding="utf-8") as f:
        f.write("\n".join(hi for sa, hi in all_pairs))

    print(f"Written {len(all_pairs)} pairs")
    print(f"SA → {args.out_src}")
    print(f"HI → {args.out_tgt}")

    if args.plot:
        orig_sizes  = [len(buckets[k]) for k in BUCKETS]
        orig_in_out = [min(len(buckets[k]), TARGET) for k in BUCKETS]
        supp_in_out = [max(0, TARGET - len(buckets[k])) for k in BUCKETS]
        labels  = ["b1\n(1-5)", "b2\n(6-10)", "b3\n(11-15)", "b4\n(16-20)", "b5\n(20+)"]
        colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
        x, w    = np.arange(5), 0.35

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.bar(x, orig_in_out, color=colors, label="Original sentences")
        ax.bar(x, supp_in_out, bottom=orig_in_out, color="#AAAAAA", alpha=0.7,
               label="Supplement (b1>=3w + b2)")
        ax.axhline(TARGET, color="black", linestyle="--", linewidth=1.5, label=f"Target ({TARGET})")
        for i, (o, s) in enumerate(zip(orig_in_out, supp_in_out)):
            ax.text(i, TARGET + 20, str(TARGET), ha="center", fontsize=10, fontweight="bold")
            ax.text(i, o / 2, str(o), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if s > 0:
                ax.text(i, o + s / 2, str(s), ha="center", va="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Sentence count", fontsize=12)
        ax.set_title(f"Bucket allocation (target = {TARGET})", fontsize=12)
        ax.legend(fontsize=10); ax.set_ylim(0, TARGET * 1.2); ax.grid(axis="y", alpha=0.3)

        ax2 = axes[1]
        ax2.bar(x - w/2, orig_sizes, w, color=colors, alpha=0.55, label="Before (original)")
        ax2.bar(x + w/2, [TARGET]*5,  w, color=colors, label="After (allocated)")
        ax2.axhline(TARGET, color="black", linestyle="--", linewidth=1.5)
        for i, v in enumerate(orig_sizes):
            ax2.text(i - w/2, v + 30, str(v), ha="center", fontsize=9, color="gray")
        for i in range(5):
            ax2.text(i + w/2, TARGET + 30, str(TARGET), ha="center", fontsize=9, fontweight="bold")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=11)
        ax2.set_ylabel("Sentence count", fontsize=12)
        ax2.set_title(f"Before vs. After  |  {TOTAL} → {TARGET*5}", fontsize=12)
        ax2.legend(fontsize=10); ax2.set_ylim(0, max(orig_sizes) * 1.15); ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Plot → {args.plot}")


if __name__ == "__main__":
    main()
