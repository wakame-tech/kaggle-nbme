import torch.nn as nn
from transformers import AutoModel
from config import Config
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


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

    @typechecked
    def forward(
        self,
        input_ids: TensorType['batch', 'token_size'],
        attention_mask: TensorType['batch', 'token_size'],
        token_type_ids: TensorType['batch', 'token_size']
    ) -> TensorType['batch', 'token_size']:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden: TensorType['batch', 'token_size', 'hidden_size'] \
            = outputs.last_hidden_state
        logits = self.fc1(hidden)
        logits = self.fc2(self.dropout(logits))
        # ここは実行時チェックされない
        logits: TensorType['batch', 'token_size', 'out': 1] \
            = self.fc3(self.dropout(logits))
        logits = logits.squeeze(-1)
        return logits
