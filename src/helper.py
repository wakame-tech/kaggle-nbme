from typing import List, Tuple
from ast import literal_eval
import pandas as pd


def parse_location(location: str) -> List[Tuple[int, int]]:
    """
    "['682 688;695 697']" -> [(682, 688), (695, 697)]
    """
    def parse(spans: str) -> str:
        loc_strs = spans.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            return (int(start), int(end))

    lst = literal_eval(location)
    return list(map(parse, lst))


def train_test_split(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert(0.0 <= test_ratio <= 1.0)
    shuffle_df = df.sample(frac=1)
    train_size = int((1.0 - test_ratio) * len(df))
    return shuffle_df[:train_size], shuffle_df[train_size:]
