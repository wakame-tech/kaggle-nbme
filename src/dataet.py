from typing import List
from qaset import QASet
from config import Config
from preprocess import preprocess
import pandas as pd


def make_dataset(config: Config) -> List[QASet]:
    features = pd.read_csv(config.features_path)
    patient_notes = pd.read_csv(config.patient_notes_path)
    train = pd.read_csv(config.train_path)
    # test = pd.read_csv(config.test_path)

    train = pd.merge(train, patient_notes, on="pn_num")
    train = pd.merge(train, features, on="feature_num")
    train = train[["id", "pn_history", "feature_text", "annotation"]]

    dataset = []
    for _, tup in list(train.iterrows()):
        text = preprocess(tup["pn_history"])
        dataset.append(QASet(text, tup["feature_text"], tup["annotation"]))
        # print(
        #     f"[ctx]:\n{tup['pn_history']}\n[question]:\n{tup['feature_text']}\n[answer]:\n{tup['annotation']}\n\n"
        # )

    return dataset
