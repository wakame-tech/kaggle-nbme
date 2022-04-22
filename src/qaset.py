
from dataclasses import dataclass


@dataclass
class QASet:
    context: str
    question: str
    answer: str
