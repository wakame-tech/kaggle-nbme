from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from transformers import BertForQuestionAnswering, BertTokenizer


@dataclass
class QASet:
    context: str
    question: str
    answer: str


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


def preprocess(text: str) -> str:
    # TODO
    return text


def make_dataset(config: Config) -> List[QASet]:
    features = pd.read_csv(config.features_path)
    patient_notes = pd.read_csv(config.patient_notes_path)
    train = pd.read_csv(config.train_path)
    # test = pd.read_csv(config.test_path)

    train = pd.merge(train, patient_notes, on="pn_num")
    train = pd.merge(train, features, on="feature_num")
    train = train[["id", "pn_history", "feature_text", "annotation"]]

    dataset = []
    for _, tup in list(train.iterrows())[:10]:
        text = preprocess(tup["pn_history"])
        dataset.append(QASet(text, tup["feature_text"], tup["annotation"]))
        # print(
        #     f"[ctx]:\n{tup['pn_history']}\n[question]:\n{tup['feature_text']}\n[answer]:\n{tup['annotation']}\n\n"
        # )

    return dataset


def predict(
    model, tokenizer, input_ids, context: str, question: str
) -> List[Tuple[int, int]]:
    """
    returns spans of prediction
    """
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    # print(f"question: 0..{sep_idx}, answer: {sep_idx}..{len(input_ids)}")
    segment_ids = [0] * (sep_idx + 1) + [1] * (len(input_ids) - sep_idx - 1)

    # output.start_logits: torch.Tensor[1, len(input_ids)]
    # output.end_logits: torch.Tensor[1, len(input_ids)]
    output = model(
        torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])
    )

    # print(f"{'token':10} {'id':5} start_logit end_logit")
    # for i, (tok, id, start_logit, end_logit) in enumerate(
    #     zip(tokens, input_ids, output.start_logits[0], output.end_logits[0])
    # ):
    #     print(f"[{i}] {tok:10} {id:5} {start_logit:.2f} {end_logit:.2f}")

    start, end = torch.argmax(output.start_logits[:, sep_idx:]), torch.argmax(
        output.end_logits[:, sep_idx:]
    )
    start, end = int(start.item()), int(end.item())
    return [(start, end)]


def slice(tokens: List[str], start: int, end: int) -> str:
    toks = tokens[start:end]
    return " ".join(tok for tok in toks)


if __name__ == "__main__":
    config = Config()
    train = make_dataset(config)

    model = BertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    for qa in train:
        # TODO
        question = f"what {qa.question}?"
        input_ids = tokenizer.encode(question, qa.context)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        spans = predict(model, tokenizer, input_ids, qa.context, qa.question)
        start, end = spans[0]
        print(f"span: {start}..{end}")
        answer = slice(tokens, start, end)
        print(f"question: {question}\nanswer: {answer}\n")
