from config import Config
import torch
import pandas as pd
from transformers import AutoTokenizer
from dataset import make_test_dataset, tokenize_and_add_labels
from model import QAModel
import numpy as np


def predict(config: Config):
    """
    TODO: dump submission.csv 
    """
    model_path = 'model.pth'
    model = QAModel(config)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    DEVICE = torch.device(config.device)
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    test = make_test_dataset(config)
    data = test.iloc[0]

    tokens = tokenizer(
        data["feature_text"],
        data["pn_history"],
        # 文の2番目の切り捨てを行う
        truncation="only_second",
        max_length=config.token_size,
        # 最大長でpaddingする
        padding='max_length',
        return_token_type_ids=True,
        return_offsets_mapping=True
    )
    # batch-rize ['token_size'] -> [1, 'token_size']
    input_ids = torch.Tensor(np.array(tokens["input_ids"])) \
        .unsqueeze(0).to(DEVICE)
    attention_mask = torch.Tensor(
        np.array(tokens["attention_mask"])).unsqueeze(0).to(DEVICE)
    token_type_ids = torch.Tensor(
        np.array(tokens["token_type_ids"])).unsqueeze(0).to(DEVICE)

    # FIXME:
    # RuntimeError: Expected tensor for argument #1 'indices'
    # to have one of the following scalar types: Long, Int;
    # but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    logits = model(input_ids, attention_mask, token_type_ids)
    print(logits)
