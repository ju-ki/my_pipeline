import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        self.df = df
        self.config = config
        self.sentence = self.df[self.config.sentence_col].values
        self.target = self.df[self.config.target_col].values
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