import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


def collate_fn(batch):
    # ELECTRA typically uses 0 for [PAD], updating mask logic for safety
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([int(x["labels"]) for x in batch])

    # Create attention mask: 1 for real tokens, 0 for padding
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels, "attention_mask": attention_mask}


def tokenizes(examples, tokenizer):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")


class NLIConfig(PretrainedConfig):
    model_type = "NLI"

    def __init__(self, vocab_size=30522, hidden_size=256, nclass=3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass


class NLI(PreTrainedModel):
    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        # Load ELECTRA-small weights
        self.bert = AutoModel.from_pretrained(
            "google/electra-small-discriminator", use_safetensors=True)

        # Classifier head (Automatically uses hidden_size 256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, config.nclass)
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, lengths=None, labels=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # ELECTRA uses the first token ([CLS]) for sentence representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
