from dataclasses import dataclass
from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm

from config import Config
from model import QAModel


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def eval_model(
    epoch: int,
    config: Config,
    model: QAModel,
    dataloader: DataLoader,
    criterion,
    writer,
):
    """
    how to evaluation
    https://www.kaggle.com/c/nbme-score-clinical-patient-notes/overview/evaluation
    """
    DEVICE = torch.device(config.device)
    model.eval()

    valid_loss = []
    preds = []
    offsets = []
    seq_ids = []
    valid_labels = []

    for i, batch in enumerate(tqdm(dataloader)):
        (
            input_ids,
            attention_mask,
            token_type_ids,
            labels,
            offset_mapping,
            sequence_ids,
        ) = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits: TensorType["batch_size", "token_size"] = model(
            input_ids, attention_mask, token_type_ids
        )

        loss = criterion(logits, labels)
        loss = torch.masked_select(loss, labels > -1.0).mean()

        valid_loss = loss.item() * input_ids.size(0)
        writer.add_scalar("valid/loss", valid_loss, epoch * len(dataloader) + i)

        preds.append(logits.detach().cpu().numpy())
        offsets.append(offset_mapping.numpy())
        seq_ids.append(sequence_ids.numpy())
        valid_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)

    location_preds = get_location_predictions(
        preds, offsets, seq_ids, config.span_thres
    )
    score = calculate_char_cv(location_preds, offsets, seq_ids, valid_labels)
    return score


def get_location_predictions(
    # NdArray['test_size', 'token_size']
    preds,
    # NdArray['test_size', 'token_size', 2]
    # ???????????? [??????index, ??????index]
    offset_mapping,
    # TensorType['test_size', 'token_size']
    # question: 0, context: 1, otherwize: nan
    sequence_ids,
    thres: float = 0.5,
) -> List[List[Tuple[int, int]]]:
    """
    preds -> spans: List['test_size', 'span_count']
    """
    all_spans: List[List[Tuple[int, int]]] = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        # logits??????probability?????????
        pred = 1 / (1 + np.exp(-pred))

        start_idx = None
        end_idx = None
        current_preds: List[Tuple[int, int]] = []
        # print(pred, offsets, seq_ids)
        for pr, offset, seq_id in zip(pred, offsets, seq_ids):
            if seq_id is None or seq_id == 0:
                continue

            if pr > thres:
                if start_idx is None:
                    start_idx = offset[0]
                end_idx = offset[1]
            elif start_idx is not None:
                current_preds.append((start_idx, end_idx))
                start_idx = None
        all_spans.append(current_preds)

    return all_spans


def calculate_char_cv(
    predictions: List[List[Tuple[int, int]]],
    # NdArray['test_size', 'token_size', 2]
    offset_mapping: np.ndarray,
    # NdArray['test_size', 'token_size']
    sequence_ids: np.ndarray,
    # NdArray['test_size', 'token_size']
    labels: np.ndarray,
) -> Metrics:
    """
    ???????????????????????????????????????
    """
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels in zip(
        predictions, offset_mapping, sequence_ids, labels
    ):
        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros(num_chars)

        # ?????????
        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0] : o[1]] = 1

        char_preds = np.zeros(num_chars)

        # ????????????
        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", labels=np.unique(all_preds)
    )
    accuracy: float = accuracy_score(all_labels, all_preds)

    return Metrics(accuracy, precision, recall, f1)
