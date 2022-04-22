from torch.utils.data import DataLoader
from model import QAModel
from helper import train_test_split
from dataset import QADataset, make_dataset
from transformers import AutoTokenizer
from config import Config
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
from itertools import chain
from typing import List
from tqdm import tqdm


def eval_model(
    config: Config,
    model: QAModel,
    dataloader: DataLoader,
    criterion
):
    DEVICE = config.device
    model.eval()
    valid_loss = []
    preds = []
    offsets = []
    seq_ids = []
    valid_labels = []

    for batch in tqdm(dataloader):
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        token_type_ids = batch[2].to(DEVICE)
        labels = batch[3].to(DEVICE)
        offset_mapping = batch[4]
        sequence_ids = batch[5]

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss = torch.masked_select(loss, labels > -1.0).mean()
        valid_loss.append(loss.item() * input_ids.size(0))

        preds.append(logits.detach().cpu().numpy())
        offsets.append(offset_mapping.numpy())
        seq_ids.append(sequence_ids.numpy())
        valid_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    location_preds = get_location_predictions(
        preds, offsets, seq_ids, test=False)
    score = calculate_char_cv(location_preds, offsets, seq_ids, valid_labels)

    return sum(valid_loss)/len(valid_loss), score


# token単位の予測結果を文字単位の範囲に変換する．
def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        pred = 1 / (1 + np.exp(-pred))  # logitsからprobabilityを計算

        start_idx = None
        end_idx = None
        current_preds = []
        for pred, offset, seq_id in zip(pred, offsets, seq_ids):
            if seq_id is None or seq_id == 0:
                continue

            # probability > 0.5 が連続している部分をtoken単位で探し，
            # そのstart位置とstop位置の（文字単位での）idxを記録する．
            if pred > 0.5:
                if start_idx is None:
                    start_idx = offset[0]
                end_idx = offset[1]
            elif start_idx is not None:
                if test:
                    current_preds.append(f"{start_idx} {end_idx}")
                else:
                    current_preds.append((start_idx, end_idx))
                start_idx = None
        if test:
            all_predictions.append("; ".join(current_preds))
        else:
            all_predictions.append(current_preds)

    return all_predictions


# 文字単位で評価値を計算する．
def calculate_char_cv(predictions, offset_mapping, sequence_ids, labels):
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):

        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros(num_chars)

        # ラベル
        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0]:o[1]] = 1

        char_preds = np.zeros(num_chars)

        # 予測結果
        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    results = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", labels=np.unique(all_preds))
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        "Accuracy": accuracy,
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }


def train_model(
    config: Config,
    model: QAModel,
    dataloader: DataLoader,
    optimizer,
    criterion
) -> float:
    model.train()
    train_loss = []

    DEVICE = config.device

    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        token_type_ids = batch[2].to(DEVICE)
        labels = batch[3].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        loss = torch.masked_select(loss, labels > -1.0).mean()
        train_loss.append(loss.item() * input_ids.size(0))
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # it's also improve f1 accuracy slightly
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return sum(train_loss)/len(train_loss)


def train(config: Config):
    df = make_dataset(config)
    train, test = train_test_split(df, 0.3)
    print(f'train: {len(train)}, test: {len(test)}')
    model = QAModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    train_dataset = QADataset(train, tokenizer, config)
    test_dataset = QADataset(test, tokenizer, config)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    epochs = 3
    train_losses: List[float] = []
    valid_losses: List[float] = []
    score_data_list = []

    for i in range(epochs):
        print(f"epoch: {i + 1}/{epochs}")

        train_loss = train_model(
            config, model, train_dataloader, optimizer, criterion)
        train_losses.append(train_loss)
        print(f"train loss: {train_loss}")

        valid_loss, score = eval_model(
            config, model, test_dataloader, criterion)
        valid_losses.append(valid_loss)
        score_data_list.append(score)
        print(f"Valid loss: {valid_loss}")
        print(f"Valid score: {score}")
