from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from tqdm import tqdm
import torch

model_dir = "./nllb_finetuned_sanhin65k"
dataset_path = "./data/hf_dataset"
output_file = "./output/translated_test_1452.txt"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]

translated_sentences = []
for example in tqdm(test_dataset, desc="Translating"):
    source_text = example["source"]
    inputs = tokenizer(source_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_sentences.append(translated_text)

with open(output_file, "w", encoding="utf-8") as f:
    for line in translated_sentences:
        f.write(line.strip() + "\n")

print(f"Translation completed. Output saved to: {output_file}")
