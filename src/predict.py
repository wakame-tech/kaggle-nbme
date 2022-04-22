from typing import List
import torch
from transformers import BertForQuestionAnswering, BertTokenizer


def predict(
    model: BertForQuestionAnswering,
    tokenizer: BertTokenizer,
    input_ids: torch.Tensor,
) -> List[List[int]]:
    """
    returns spans of token_ids
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
    return [input_ids[start:end]]
