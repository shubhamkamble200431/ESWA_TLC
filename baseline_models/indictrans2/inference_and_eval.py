import torch
import evaluate
import nltk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from IndicTransToolkit import IndicProcessor

from comet import download_model, load_from_checkpoint

nltk.download("wordnet")
nltk.download("omw-1.4")

MODEL_PATH = "./final_san_hin_model"

BASE_MODEL = "ai4bharat/indictrans2-indic-indic-dist-320M"

TEST_SRC_FILE = "./data/test.san_Deva"
TEST_TGT_FILE = "./data/test.hin_Deva"

OUTPUT_FILE = "./data/predictions.txt"

print("Loading evaluation metrics...")

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
ter_metric = evaluate.load("ter")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")

print("Loading COMET model...")

comet_model_path = download_model("Unbabel/wmt22-comet-da")

comet_model = load_from_checkpoint(comet_model_path)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Loading fine-tuned model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).cuda()

model.eval()

ip = IndicProcessor(inference=True)

SRC_LANG = "san_Deva"
TGT_LANG = "hin_Deva"

with open(TEST_SRC_FILE, encoding="utf-8") as f:
    test_src = f.read().splitlines()

with open(TEST_TGT_FILE, encoding="utf-8") as f:
    test_tgt = f.read().splitlines()

assert len(test_src) == len(test_tgt)

print(f"Loaded {len(test_src)} test sentences")

BATCH_SIZE = 32

all_predictions = []

print("Generating translations...")

for i in range(0, len(test_src), BATCH_SIZE):

    batch_sentences = test_src[i:i+BATCH_SIZE]

    batch = ip.preprocess_batch(
        batch_sentences,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG
    )

    inputs = tokenizer(
        batch,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():

        generated_tokens = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    translations = ip.postprocess_batch(
        decoded,
        lang=TGT_LANG
    )

    cleaned_translations = []

    for text in translations:

        text = text.replace("hin_Deva", "")
        text = text.replace("san_Deva", "")

        text = " ".join(text.split())

        cleaned_translations.append(text.strip())

    all_predictions.extend(cleaned_translations)

    print(f"Processed {min(i+BATCH_SIZE, len(test_src))}/{len(test_src)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for pred in all_predictions:
        f.write(pred + "\n")

print(f"\nPredictions saved to: {OUTPUT_FILE}")

references = [[ref] for ref in test_tgt]

bleu = bleu_metric.compute(
    predictions=all_predictions,
    references=references
)

chrf = chrf_metric.compute(
    predictions=all_predictions,
    references=references
)

chrf2 = chrf_metric.compute(
    predictions=all_predictions,
    references=references,
    word_order=2
)

ter = ter_metric.compute(
    predictions=all_predictions,
    references=references
)

meteor = meteor_metric.compute(
    predictions=all_predictions,
    references=test_tgt
)

bertscore = bertscore_metric.compute(
    predictions=all_predictions,
    references=test_tgt,
    lang="hi"
)

avg_precision = sum(bertscore["precision"]) / len(bertscore["precision"])

avg_recall = sum(bertscore["recall"]) / len(bertscore["recall"])

avg_f1 = sum(bertscore["f1"]) / len(bertscore["f1"])

print("\nComputing COMET Score...")

comet_data = []

for src, mt, ref in zip(test_src, all_predictions, test_tgt):

    comet_data.append({
        "src": src,
        "mt": mt,
        "ref": ref
    })

comet_output = comet_model.predict(
    comet_data,
    batch_size=8,
    gpus=1
)

comet_score = comet_output.system_score

print("\n======================================")
print(" MACHINE TRANSLATION EVALUATION")
print("======================================")

print(f"BLEU                 : {bleu['score']:.2f}")

print(f"CHRF2                : {chrf['score']:.2f}")

print(f"CHRF2++              : {chrf2['score']:.2f}")

print(f"TER                  : {ter['score']:.2f}")

print(f"METEOR               : {meteor['meteor']:.4f}")

print(f"BERTScore Precision  : {avg_precision:.4f}")

print(f"BERTScore Recall     : {avg_recall:.4f}")

print(f"BERTScore F1         : {avg_f1:.4f}")

print(f"COMET                : {comet_score:.4f}")

print("\n======================================")
print(" SAMPLE TRANSLATIONS")
print("======================================")

for i in range(min(5, len(test_src))):

    print("\nSOURCE:")
    print(test_src[i])

    print("\nREFERENCE:")
    print(test_tgt[i])

    print("\nPREDICTION:")
    print(all_predictions[i])

    print("\n--------------------------------------")
