import pandas as pd
from typing import List
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def get_tokenizer(config):
    assert hasattr(config, "model_name"), "Please create model_name(string roberta-base) attributes"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer


def get_max_lengths(df: pd.DataFrame, cols: List[str], config=None):
    lengths_dict = {}
    for text_col in cols:
        lengths = []
        tk0 = tqdm(df[text_col].fillna("").values, total=len(df))
        for text in tk0:
            length = len(config.tokenizer(text, add_special_tokens=False)['input_ids'])
            lengths.append(length)
        lengths_dict[text_col] = lengths
    max_lengths = 0
    for text_col in cols:
        max_lengths += max(lengths_dict[text_col])
    return max_lengths