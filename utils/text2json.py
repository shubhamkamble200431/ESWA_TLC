import json
import argparse


def txt_to_json(src_file, tgt_file, output_json):
    with open(src_file, "r", encoding="utf-8") as sf, \
         open(tgt_file, "r", encoding="utf-8") as tf:
        src_lines = sf.readlines()
        tgt_lines = tf.readlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: src={len(src_lines)}, tgt={len(tgt_lines)}")

    data = [
        {"translation": {"sa": s.strip(), "hi": t.strip()}}
        for s, t in zip(src_lines, tgt_lines)
    ]

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

    print(f"JSON written: {output_json}  ({len(data)} pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parallel .sa/.hi text files to JSON")
    parser.add_argument("--src",  required=True, help="Sanskrit source file (.sa)")
    parser.add_argument("--tgt",  required=True, help="Hindi target file (.hi)")
    parser.add_argument("--out",  required=True, help="Output JSON path")
    args = parser.parse_args()
    txt_to_json(args.src, args.tgt, args.out)
