from dataclasses import dataclass
from pathlib import Path
from typing import List

from transformers import BertForQuestionAnswering, BertTokenizer
from config import Config
from dataet import make_dataset
from predict import predict

if __name__ == "__main__":
    config = Config()
    train = make_dataset(config)

    model = BertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    for qa in train[:5]:
        # TODO
        question = f"what {qa.question}?"
        input_ids = tokenizer.encode(question, qa.context)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        spans = predict(model, tokenizer, input_ids)
        span1 = spans[0]
        print(f"span: {span1}")
        print(tokenizer.decode(span1))
