from torch.utils.data import DataLoader
from model import QAModel
from helper import train_test_split
from dataset import QADataset, make_dataset
from transformers import AutoTokenizer
from config import Config
from torch import optim
import torch.nn as nn
import torch
from typing import List, Tuple
from tqdm import tqdm
from eval import eval_model


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
        input_ids, attention_mask, token_type_ids, labels, offset_mapping, sequence_ids = batch

        # [batch, token_size]
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        loss = torch.masked_select(loss, labels > -1.0).mean()
        train_loss.append(loss.item() * input_ids.size(0))
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # it's also improve f1 accuracy slightly
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return sum(train_loss) / len(train_loss)


def train(config: Config):
    """
    train & save model 
    """
    df = make_dataset(config)
    train, test = train_test_split(df, 0.3)
    print(f'train: {len(train)}, test: {len(test)}')
    DEVICE = torch.device(config.device)
    model = QAModel(config).to(DEVICE)
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

    train_losses: List[float] = []
    valid_losses: List[float] = []

    for i in range(config.epochs):
        print(f"epoch: {i + 1}/{config.epochs}")

        train_loss = train_model(
            config, model, train_dataloader, optimizer, criterion)
        train_losses.append(train_loss)
        print(f"train/loss: {train_loss}")

        valid_loss, metrics = eval_model(
            config, model, test_dataloader, criterion)
        valid_losses.append(valid_loss)
        print(f"valid/loss: {valid_loss}")
        print(f"valid/metrics: {metrics}")

    torch.save(model.to('cpu').state_dict(), 'model.pth')
