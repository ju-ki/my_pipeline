from transformer import AutoTokenizer


def get_tokenizer(config):
    assert hasattr(config, "model_name"), "Please create model_name(string roberta-base) attributes"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer