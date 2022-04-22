import torch.nn as nn
from transformers import AutoModel
from config import Config


class QAModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model)
        dropout = 0.2
        self.dropout = nn.Dropout(p=dropout)
        self.config = config
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.fc1(outputs[0])
        logits = self.fc2(self.dropout(logits))
        logits = self.fc3(self.dropout(logits)).squeeze(-1)
        return logits
