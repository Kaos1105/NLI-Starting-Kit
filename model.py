import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig


# collate_fn is used by the Trainer to batch input dictionaries.
# It converts a list of examples into tensors and pads the batch
# by using the existing input length information.
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([int(x["labels"]) for x in batch])
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}


# This helper wraps a tokenizer call for fixed-length tokenization.
# It is compatible with the existing test.py text-format input.
def tokenizes(examples, tokenizer):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")


class NLIConfig(PretrainedConfig):
    model_type = "NLI"

    def __init__(self, vocab_size=20000, hidden_size=512, nclass=3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass


class NLI(PreTrainedModel):
    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layer converts token ids to dense vectors.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Bidirectional LSTM captures both forward and backward context.
        # We use hidden_size // 2 because bidirectional doubles the output dimension.
        # num_layers=2 enables recurrent dropout between layers.
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Small classifier head gives the model extra capacity after sequence encoding.
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.nclass),
        )

        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, lengths, labels=None, **kwargs):
        # 1. Convert token ids to embeddings for the whole batch.
        x = self.embedding(input_ids)

        # 2. Pack the padded sequence so LSTM ignores padding positions.
        packed = pack_padded_sequence(
            x,
            lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )

        # 3. Run the bidirectional LSTM.
        _, (h, _) = self.lstm(packed)

        # h contains the final states for both directions.
        # We concatenate forward and backward states to make a single vector.
        h = torch.cat((h[-2], h[-1]), dim=-1)

        # 4. Classify the final sentence representation into 3 labels.
        logits = self.classifier(h)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
