import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


def collate_fn(batch):
    # test.py expects 'lengths', so we provide it even if the Transformer doesn't strictly need it
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([int(x["labels"]) for x in batch])
    # Create attention mask (1 for real tokens, 0 for [PAD])
    # Assuming [PAD] token ID is 0 or 3 based on your original tokenizer
    attention_mask = (input_ids != 3).long()
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels, "attention_mask": attention_mask}


def tokenizes(examples, tokenizer):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")


class NLIConfig(PretrainedConfig):
    model_type = "NLI"

    def __init__(self, vocab_size=30522, hidden_size=512, nclass=3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass


class NLI(PreTrainedModel):
    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        # Load the architecture and weights of a pre-trained small BERT
        self.bert = AutoModel.from_pretrained(
            "prajjwal1/bert-small", use_safetensors=True)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, config.nclass)
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, lengths=None, labels=None, attention_mask=None, **kwargs):
        # Generate mask if not provided (needed for compatibility with test.py)
        if attention_mask is None:
            attention_mask = (input_ids != 3).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
