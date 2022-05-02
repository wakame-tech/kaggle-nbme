
def preprocess_question(text: str) -> str:
    # TODO
    return text.replace("-OR-", "; ").replace("-", " ")
