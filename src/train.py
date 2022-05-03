from typing import List, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from dataset import QADataset, make_dataset
from eval import eval_model
from helper import train_test_split
from model import QAModel


def train_model(
    epoch: int,
    config: Config,
    model: QAModel,
    dataloader: DataLoader,
    optimizer,
    criterion,
    writer,
):
    model.train()
    train_loss = []

    DEVICE = config.device

    for i, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        (
            input_ids,
            attention_mask,
            token_type_ids,
            labels,
            offset_mapping,
            sequence_ids,
        ) = batch

        # [batch, token_size]
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        loss = torch.masked_select(loss, labels > -1.0).mean()

        train_loss = loss.item() * input_ids.size(0)
        writer.add_scalar("train/loss", train_loss, epoch * len(dataloader) + i)

        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # it's also improve f1 accuracy slightly
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def train(config: Config):
    """
    train & save model
    """
    df = make_dataset(config)
    train, test = train_test_split(df, 0.3)
    print(f"train: {len(train)}, test: {len(test)}")
    DEVICE = torch.device(config.device)
    model = QAModel(config).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    train_dataset = QADataset(train, tokenizer, config)
    test_dataset = QADataset(test, tokenizer, config)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    writer = SummaryWriter()

    for epoch in range(config.epochs):
        print(f"epoch: {epoch + 1}/{config.epochs}")

        train_model(epoch, config, model, train_dataloader, optimizer, criterion, writer)
        metrics = eval_model(epoch, config, model, test_dataloader, criterion, writer)
        writer.add_scalar("valid/f1", metrics.f1, epoch * len(test_dataloader))
        writer.add_scalar('valid/acc', metrics.accuracy, epoch * len(test_dataloader))
        writer.add_scalar('valid/prec', metrics.precision, epoch * len(test_dataloader))

    torch.save(model.to("cpu").state_dict(), "model.pth")
