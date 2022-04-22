from config import Config
from transformers import AutoTokenizer
from dataset import QADataset, make_dataset
from helper import train_test_split
from model import QAModel
from torch.utils.data import DataLoader


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
