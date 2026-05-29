from datasets import DatasetDict, Dataset
import os

data_path = "./data"

def load_text_data(lang):
    def read_file(filename):
        path = os.path.join(data_path, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        else:
            print(f"Warning: {filename} not found!")
            return []

    return {
        "train": read_file(f"train.{lang}"),
        "validation": read_file(f"dev.{lang}"),
        "test": read_file(f"test.{lang}"),
    }

maithili_data = load_text_data("san_Deva")
hindi_data = load_text_data("hin_Deva")

assert len(maithili_data["train"]) == len(hindi_data["train"]), "Mismatch in training data length"
assert len(maithili_data["validation"]) == len(hindi_data["validation"]), "Mismatch in validation data length"
assert len(maithili_data["test"]) == len(hindi_data["test"]), "Mismatch in test data length"

dataset = DatasetDict({
    "train": Dataset.from_dict({"source": maithili_data["train"], "target": hindi_data["train"]}),
    "validation": Dataset.from_dict({"source": maithili_data["validation"], "target": hindi_data["validation"]}),
    "test": Dataset.from_dict({"source": maithili_data["test"], "target": hindi_data["test"]}),
})

hf_dataset_path = "./data/hf_dataset"
dataset.save_to_disk(hf_dataset_path)

print(f"Dataset saved successfully at {hf_dataset_path}!")
