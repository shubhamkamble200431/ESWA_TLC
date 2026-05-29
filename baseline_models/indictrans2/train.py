import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from IndicTransToolkit import IndicProcessor

model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,

)

ip = IndicProcessor(inference=False)

SRC_LANG = "san_Deva"
TGT_LANG = "hin_Deva"


def load_parallel(src_file, tgt_file):

    with open(src_file, encoding="utf-8") as f:
        src = f.read().splitlines()

    with open(tgt_file, encoding="utf-8") as f:
        tgt = f.read().splitlines()

    return Dataset.from_dict({
        "src": src,
        "tgt": tgt
    })

train_dataset = load_parallel(
    "./data/train.san_Deva",
    "./data/train.hin_Deva"
)

valid_dataset = load_parallel(
    "./data/dev.san_Deva",
    "./data/dev.hin_Deva"
)

MAX_LEN = 128

def preprocess(batch):

    inputs = ip.preprocess_batch(
        batch["src"],
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG
    )

    targets = ip.preprocess_batch(
        batch["tgt"],
        src_lang=TGT_LANG,
        tgt_lang=TGT_LANG
    )

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=["src", "tgt"]
)

valid_dataset = valid_dataset.map(
    preprocess,
    batched=True,
    remove_columns=["src", "tgt"]
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./indictrans2_san_hin",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    dataloader_num_workers=8,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    report_to="tensorboard"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("./final_san_hin_model")
tokenizer.save_pretrained("./final_san_hin_model")
