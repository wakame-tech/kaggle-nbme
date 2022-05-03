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
    # id, location
    submission_path: Path = Path("dataset/submission.csv")
    # model
    # model: str = "bert-base-uncased"
    model: str = 'microsoft/deberta-base'
    # token size
    token_size: int = 128 # 416
    # batch size (default: 8)
    batch_size: int = 8
    # device: 'cpu' or 'cuda'
    device: str = "cuda"
    # device: str = 'cpu'
    # span thres
    span_thres: float = 0.4
    # epochs
    epochs: int = 10
    # debug?
    debug: bool = True
