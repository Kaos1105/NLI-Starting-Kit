import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer, processors
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, EarlyStoppingCallback

from datasets import load_dataset, Dataset

from model import *

import evaluate

# Load the MultiNLI dataset from Hugging Face.
# This dataset contains premise/hypothesis pairs and 3-way labels.
mnli = load_dataset("nyu-mll/multi_nli")

df_train = mnli['train'].to_pandas()
df_train = df_train.dropna()

df_val = mnli['validation_matched'].to_pandas()
df_val = df_val.dropna()

# Build a word-level tokenizer from the training text.
# We train on both premises and hypotheses to cover all tokens.
Sentences = df_train["premise"].to_list()
Sentences.extend(df_train["hypothesis"].to_list())

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
# Tokenizer post-processing defines [CLS], [SEP], [PAD], and [MASK] tokens.
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[UNK]", 0),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("[PAD]", 3),
        ("[MASK]", 4)
    ],
)

trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2,
)

# Train the tokenizer on all sentences from the training set.
tokenizer.train_from_iterator(
    Sentences, trainer=trainer, length=len(Sentences)
)

# Wrap the trained tokenizer in a Hugging Face fast tokenizer.
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Save the tokenizer under MODEL/ so it can be loaded later by test.py.
hf_tokenizer.save_pretrained("./MODEL")


# Data collator for the Trainer: turns batches into tensors.
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}


# The tokenizer expects a single string formatted in the same way as test.py.
def tokenizes(examples):
    return hf_tokenizer(examples, truncation=True, max_length=128, padding="max_length")


# Compute standard validation metrics after each evaluation.
def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}
    metric = evaluate.load("accuracy")
    res["accuracy"] = metric.compute(
        predictions=pred, references=targ
    )["accuracy"]
    metric = evaluate.load("precision")
    res["precision"] = metric.compute(
        predictions=pred, references=targ, average="macro", zero_division=0
    )["precision"]
    metric = evaluate.load("recall")
    res["recall"] = metric.compute(
        predictions=pred, references=targ, average="macro", zero_division=0
    )["recall"]
    metric = evaluate.load("f1")
    res["f1"] = metric.compute(
        predictions=pred, references=targ, average="macro"
    )["f1"]
    return res


# Use the same input format as test.py, combining premise and hypothesis
# with the [CLS] separator token in between.
tokenized_train = (df_train["premise"] + " [CLS] " +
                   df_train["hypothesis"]).apply(tokenizes)
train_set = Dataset.from_dict(
    {
        "input_ids": [t["input_ids"] for t in tokenized_train],
        "labels": df_train["label"].astype(int).tolist(),
    }
)

tokenized_val = (df_val["premise"] + " [CLS] " +
                 df_val["hypothesis"]).apply(tokenizes)
val_set = Dataset.from_dict(
    {
        "input_ids": [t["input_ids"] for t in tokenized_val],
        "labels": df_val["label"].astype(int).tolist(),
    }
)

# Instantiate the model with the trained tokenizer vocabulary size.
# Reduce hidden_size to lower GPU memory usage.
model = NLI(NLIConfig(vocab_size=len(
    hf_tokenizer.get_vocab()), hidden_size=512))

allparams = sum(p.numel() for p in model.parameters())
trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("All Param:", allparams, "Train Params:", trainparams)

# TrainingArguments control the Hugging Face Trainer behavior.
# We use a lower learning rate and evaluation at each epoch.
args = TrainingArguments(
    output_dir="./NLIMODEL",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    dataloader_pin_memory=True,
    # Reduce batch sizes to lower GPU memory usage.
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    # Use gradient accumulation to maintain effective batch size.
    gradient_accumulation_steps=2,
    # Enable mixed precision (FP16) for ~2x speed and lower memory.
    fp16=True,
    learning_rate=1e-3,
    weight_decay=0.01,
    # Reduce epochs to finish faster.
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model and save the final weights and tokenizer.
trainer.train()
model.save_pretrained("./MODEL")
