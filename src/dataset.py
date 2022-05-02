from transformers import AutoTokenizer
from helper import parse_location
from config import Config
from preprocess import preprocess_question
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def tokenize_and_add_labels(
    tokenizer: AutoTokenizer,
    data: pd.Series,
    config: Config,
) -> pd.DataFrame:
    out = tokenizer(
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
    # input_ids: トークンIDのリスト
    # token_type_ids: 文の種類を表すマスク(0, 1)
    labels = [0.0] * len(out["input_ids"])
    out["sequence_ids"] = out.sequence_ids()

    for idx, (seq_id, offsets) in enumerate(zip(out["sequence_ids"], out["offset_mapping"])):
        is_test_data = 'location' not in data.index
        if not seq_id or seq_id == 0 or is_test_data:
            labels[idx] = -1
            continue

        token_start, token_end = offsets
        for feature_start, feature_end in data["location"]:
            if token_start >= feature_start and token_end <= feature_end:
                labels[idx] = 1.0
                break

    out["labels"] = labels
    return out


def loopup(df, ref_df, key_label: str, value_label: str) -> pd.DataFrame:
    df[value_label] = df.merge(
        ref_df[[key_label, value_label]], on=key_label, how='right')[value_label]
    return df


def join_dfs(df, features, patient_notes):
    df = loopup(df, features, 'feature_num', 'feature_text')
    df = loopup(df, patient_notes, 'pn_num', 'pn_history')
    return df


def make_test_dataset(config: Config) -> pd.DataFrame:
    features = pd.read_csv(config.features_path)
    patient_notes = pd.read_csv(config.patient_notes_path)
    test = pd.read_csv(config.test_path)

    test = join_dfs(test, features, patient_notes)
    test = test[["id", "pn_history", "feature_text"]]
    return test


def make_dataset(config: Config) -> pd.DataFrame:
    features = pd.read_csv(config.features_path)
    patient_notes = pd.read_csv(config.patient_notes_path)
    train = pd.read_csv(config.train_path)

    train = join_dfs(train, features, patient_notes)
    train['location'] = train['location'].apply(parse_location)
    train = train[["id", "pn_history",
                   "feature_text", "annotation", 'location']]
    train['feature_text'] = train['feature_text'].apply(preprocess_question)

    print(train.head())
    if config.debug:
        return train.iloc[:10, :]
    else:
        return train


class QADataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, config: Config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        tokens = tokenize_and_add_labels(self.tokenizer, data, self.config)

        input_ids = np.array(tokens["input_ids"])
        attention_mask = np.array(tokens["attention_mask"])
        token_type_ids = np.array(tokens["token_type_ids"])

        labels = np.array(tokens["labels"])
        offset_mapping = np.array(tokens['offset_mapping'])
        sequence_ids = np.array(tokens['sequence_ids']).astype("float16")

        return input_ids, attention_mask, token_type_ids, labels, offset_mapping, sequence_ids
