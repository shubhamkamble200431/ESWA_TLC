import os
import glob
import argparse


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def convert_txt_to_tsv(txt_path, src_lines, ref_lines):
    hyp_lines = read_lines(txt_path)
    n_src, n_ref, n_hyp = len(src_lines), len(ref_lines), len(hyp_lines)
    if not (n_src == n_ref == n_hyp):
        print(f"  Line count mismatch in {os.path.basename(txt_path)}: "
              f"src={n_src}, ref={n_ref}, hyp={n_hyp}. "
              f"Will process min({n_src}, {n_ref}, {n_hyp}) lines.")
    n_lines = min(n_src, n_ref, n_hyp)
    tsv_path = os.path.splitext(txt_path)[0] + "_linked.tsv"
    with open(tsv_path, "w", encoding="utf-8") as out:
        out.write("src_len\treference\thypothesis\n")
        for i in range(n_lines):
            src = src_lines[i]
            ref = ref_lines[i].replace("\t", " ")
            hyp = hyp_lines[i].replace("\t", " ")
            src_len = len(src.split())
            out.write(f"{src_len}\t{ref}\t{hyp}\n")
    return tsv_path


def main():
    parser = argparse.ArgumentParser(description="Convert NMT prediction .txt files to .tsv")
    parser.add_argument("--pred_dir",  required=True, help="Directory with .txt prediction files")
    parser.add_argument("--src_file",  required=True, help="Source file (.sa)")
    parser.add_argument("--ref_file",  required=True, help="Reference file (.hi)")
    args = parser.parse_args()

    for path in (args.src_file, args.ref_file):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    src_lines = read_lines(args.src_file)
    ref_lines = read_lines(args.ref_file)
    print(f"src lines: {len(src_lines)}  ref lines: {len(ref_lines)}")

    txt_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.txt")))
    if not txt_files:
        print(f"No .txt files found in: {args.pred_dir}")
        return

    print(f"Found {len(txt_files)} .txt file(s) in {args.pred_dir}")
    for txt_path in txt_files:
        print(f"Processing: {os.path.basename(txt_path)}")
        tsv_path = convert_txt_to_tsv(txt_path, src_lines, ref_lines)
        print(f"  Saved: {tsv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
