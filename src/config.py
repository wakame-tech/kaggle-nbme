from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # feature_num, case_num, feature_text
    features_path: Path = Path("dataset/features.csv")
    # pn_num, case_num, pn_history
    patient_notes_path: Path = Path("dataset/patient_notes.csv")
    # id, case_num, pn_num, feature_num, annotation, location
    train_path: Path = Path("dataset/train.csv")
    # id, case_num, pn_num, feature_num, annotation, location
    test_path: Path = Path("dataset/test.csv")
    # model
    model: str = 'bert-base-uncased'
    # model: str = 'deepset/roberta-base-squad2'
    # token size
    token_size: int = 416
    # batch size (default: 8)
    batch_size: int = 16
    # device: 'cpu' or 'cuda'
    # device: str = 'cuda'
    device: str = 'cpu'
    # epochs
    epochs: int = 1
    # debug?
    debug: bool = True
