from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk

dataset_path = "./data/hf_dataset"
datasets = load_from_disk(dataset_path)

model_name = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        examples["target"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=500,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("./nllb_finetuned_sanhin65k")
tokenizer.save_pretrained("./nllb_finetuned_sanhin65k")

print("Fine-tuning completed and model saved!")
