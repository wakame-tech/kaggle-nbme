from typing import List, Tuple
from config import Config
import torch
import pandas as pd
from transformers import AutoTokenizer
from eval import get_location_predictions
from dataset import make_test_dataset, tokenize_and_add_labels
from model import QAModel
import numpy as np
import matplotlib.pyplot as plt


def visualize(config: Config, logits: torch.Tensor):
    """
    render probablity distribution
    """
    assert len(logits.shape) == 1
    preds = 1 / (1 + np.exp(-logits[0]))
    n = len(preds)

    plt.xlabel('token index')
    plt.ylabel('pr')
    plt.plot(range(n), preds)
    plt.hlines(config.span_thres, 0, n)
    plt.show()


def predict(
    config: Config,
    tokenizer: AutoTokenizer,
    model: QAModel,
    series: pd.Series,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    データ1つ -> 予測スパンとlogits
    """
    DEVICE = torch.device(config.device)
    tokens = tokenize_and_add_labels(tokenizer, series, config)
    # batch-rize ['token_size'] -> [1, 'token_size']
    input_ids = torch.LongTensor(np.array(tokens["input_ids"])) \
        .unsqueeze(0).to(DEVICE)
    # [1, 'token_size']
    attention_mask = torch.LongTensor(
        np.array(tokens["attention_mask"])).unsqueeze(0).to(DEVICE)
    # [1, 'token_size']
    token_type_ids = torch.LongTensor(
        np.array(tokens["token_type_ids"])).unsqueeze(0).to(DEVICE)

    # [1, 'token_size']
    logits = model(input_ids, attention_mask, token_type_ids)
    assert logits.shape == (1, config.token_size)
    # torch.Tensor -> numpy.ndarray
    logits = logits.detach().cpu().numpy()

    # [1, 'token_size']
    batched_offset_mapping = np.array(tokens['offset_mapping'])[np.newaxis, :]
    batched_sequence_ids = np.array(tokens['sequence_ids'])[np.newaxis, :]

    assert batched_offset_mapping.shape == (1, config.token_size, 2)
    assert batched_sequence_ids.shape == (1, config.token_size)

    spans = get_location_predictions(
        logits, batched_offset_mapping, batched_sequence_ids, config.span_thres)[0]
    return logits[0], spans


def test(config: Config):
    """
    TODO: dump submission.csv
    """
    test = make_test_dataset(config)

    model_path = 'model.pth'
    model = QAModel(config)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    DEVICE = torch.device(config.device)
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    submission = test['id'].to_frame()

    for idx, series in test.iterrows():
        logits, spans = predict(config, tokenizer, model, series)
        submission.loc[idx, 'location'] = ';'.join(
            map(lambda span: f'{span[0]} {span[1]}', spans)
        )
        print(spans)
        # visualize(config, logits)

    submission.to_csv(config.submission_path, index=False)
