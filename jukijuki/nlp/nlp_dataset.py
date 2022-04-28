import torch
from torch.utils.data import Dataset


class SimpleTrainDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        assert hasattr(config, "sentence_col"), "Please create sentence_col attribute"
        assert hasattr(config, "target_col"), "Please create target_col attribute"
        self.df = df
        self.config = config
        self.sentence = self.df[self.config.sentence_col].to_numpy()
        self.target = self.df[self.config.target_col].to_numpy()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        bert_sentence = self.tokenizer(
            text=sentence,
            add_special_tokens=True,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True)
        input_ids = torch.tensor(bert_sentence["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(bert_sentence["attention_mask"], dtype=torch.long)
        target = torch.tensor(self.target[idx], dtype=torch.float)
        return input_ids, attention_mask, target


class SimpleTestDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        assert hasattr(config, "sentence_col"), "Please create sentence_col attribute"
        self.df = df
        self.config = config
        self.sentence = self.df[self.config.sentence_col].to_numpy()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        bert_sentence = self.tokenizer(
            text=sentence,
            add_special_tokens=True,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True)
        input_ids = torch.tensor(bert_sentence["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(bert_sentence["attention_mask"], dtype=torch.long)
        return input_ids, attention_mask


def convert_examples_to_features(data, tokenizer, max_len,):
    data = data.replace('\n', '')
    tok = tokenizer.encode_plus(
        data,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent


class ConvertTrainDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        assert hasattr(config, "sentence_col"), "Please create sentence_col attribute"
        assert hasattr(config, "target_col"), "Please create target_col attribute"
        self.df = df
        self.config = config
        self.sentence = self.df[self.config.sentence_col].to_numpy()
        self.target = self.df[self.config.target_col].to_numpy()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        bert_sentence = convert_examples_to_features(
            data=sentence,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len
        )
        input_ids = torch.tensor(bert_sentence["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(bert_sentence["attention_mask"], dtype=torch.long)
        target = torch.tensor(self.target[idx], dtype=torch.float)
        return input_ids, attention_mask, target


class ConvertTestDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        assert hasattr(config, "sentence_col"), "Please create sentence_col attribute"
        self.df = df
        self.config = config
        self.sentence = self.df[self.config.sentence_col].to_numpy()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        bert_sentence = convert_examples_to_features(
            data=sentence,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len
        )
        input_ids = torch.tensor(bert_sentence["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(bert_sentence["attention_mask"], dtype=torch.long)
        return input_ids, attention_mask