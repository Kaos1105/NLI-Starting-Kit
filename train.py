import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from model import *

# 1. Load the MultiNLI dataset
mnli = load_dataset("nyu-mll/multi_nli")

df_train = mnli['train'].to_pandas().dropna()
df_val = mnli['validation_matched'].to_pandas().dropna()

# 2. Setup Pre-trained Tokenizer (Transfer Learning)
# Using 'bert-small' because it matches the 40M parameter constraint
model_checkpoint = "prajjwal1/bert-small"
hf_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Save the tokenizer under MODEL/ for test.py to load
hf_tokenizer.save_pretrained("./MODEL")

# 3. Define Tokenization and Dataset preparation


def tokenizes_data(text_list):
    return hf_tokenizer(
        text_list,
        truncation=True,
        max_length=128,
        padding="max_length"
    )


# test.py format: Premise + " [CLS] " + Hypothesis
# We must train with the EXACT same format as test.py
tokenized_train = (df_train["premise"] + " [CLS] " +
                   df_train["hypothesis"]).tolist()
tokenized_val = (df_val["premise"] + " [CLS] " + df_val["hypothesis"]).tolist()

train_encodings = tokenizes_data(tokenized_train)
val_encodings = tokenizes_data(tokenized_val)

train_set = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": df_train["label"].astype(int).tolist(),
})

val_set = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": df_val["label"].astype(int).tolist(),
})

# 4. Initialize Model
# We ignore the manual vocab_size because bert-small uses its own pre-trained vocab (30522)
config = NLIConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=512,  # Matches bert-small hidden size
    nclass=3
)
model = NLI(config)

# Check parameters (bert-small is ~28M)
allparams = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {allparams / 1e6:.2f}M")

# 5. Metrics


def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    acc = evaluate.load("accuracy").compute(predictions=pred, references=targ)
    return acc


# 6. Training Arguments
# CRITICAL: Learning rate must be small (2e-5) for Transfer Learning
args = TrainingArguments(
    output_dir="./NLIMODEL",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    per_device_train_batch_size=8,  # Increased for better stability
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,                     # Faster training on GPU
    learning_rate=2e-5,            # Small LR for BERT
    weight_decay=0.01,
    num_train_epochs=10,            # Transformers need fewer epochs
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate_fn,  # From model.py
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 7. Start Training
trainer.train()

# Save the model in the format test.py expects
model.save_pretrained("./MODEL")
print("Training complete. Model saved to ./MODEL")
