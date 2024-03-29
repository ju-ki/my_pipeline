import os
import torch
import hashlib
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.pipeline import Pipeline
from .bm25 import BM25Transformer
from .scdv import SCDVEmbedder
from .util import AbstractBaseBlock
tqdm.pandas()


class StringLengthBlock(AbstractBaseBlock):
    def __init__(self, cols: str):
        self.cols = cols

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.cols] = input_df[self.cols].str.len()
        return out_df.add_prefix('string_length_')


class GetCountWordBlock(AbstractBaseBlock):
    """単語数を取得するブロック"""

    def __init__(self, cols: str, text_normalize=None):
        self.cols = cols
        self.text_normalize = text_normalize

    def transform(self, input_df):
        _input_df = input_df.copy()
        out_df = pd.DataFrame()
        if self.text_normalize:
            _input_df[self.cols] = self.text_normalize(_input_df[self.cols].fillna("NaN"))
        out_df[self.cols] = _input_df[self.cols].fillna("NaN").apply(
            lambda x: len(x.split()))
        return out_df.add_prefix("count_word_")


class TfidfBlock(AbstractBaseBlock):
    def __init__(self, cols: str, n_components: int = 50, name="svd", text_normalize=None):
        """
        ref:
          https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/
        args:
         cols (str): col_name
         n_components (int): number of dimension
        """
        self.cols = cols
        self.n_components = n_components
        self.name = name
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)
        self.text_normalize = text_normalize

    def confirm_cumulative_contribution_rate(self, input_df):
        x = self.preprocess(input_df)
        x = TfidfVectorizer.fit_transform(x)
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def preprocess(self, input_df):
        if self.text_normalize:
            x = self.text_normalize(input_df[self.cols])
        else:
            x = input_df[self.cols].fillna("")
        return x

    def fit(self, input_df, y=None):
        text = self.preprocess(input_df)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            (self.name, self.decomp),
        ])

        self.pipeline.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pipeline.transform(text)
        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f"{self.cols}_tfidf_").add_suffix(f"_{self.name}_feature")


class BM25Block(AbstractBaseBlock):
    def __init__(self, cols: str, n_components: int = 50, name = "svd", text_normalize=None):
        self.cols = cols
        self.n_components = n_components
        self.name = name
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)
        self.text_normalize = text_normalize

    def confirm_cumulative_contribution_rate(self, input_df):
        x = self.preprocess(input_df)
        x = CountVectorizer().fit_transform(x)
        bm25 = BM25Transformer()
        bm25.fit(x)
        x = bm25().transform(x)
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def preprocess(self, input_df):
        if self.text_normalize:
            x = self.text_normalize(input_df[self.cols])
        else:
            x = input_df[self.cols].fillna("")
        return x

    def fit(self, input_df, y=None):
        text = self.preprocess(input_df)
        self.pipeline = Pipeline([
            ("CountVectorizer", CountVectorizer()),
            ("BM25Transformer", BM25Transformer()),
            (self.name, self.decomp)
        ])

        self.pipeline.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pipeline.transform(text)
        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f"{self.cols}_bm25_").add_suffix(f"_{self.name}_feature")


class Doc2VecBlock(AbstractBaseBlock):
    def __init__(self, cols: str, params=None, text_normalize=None):
        self.cols = cols
        self.text_normalize = text_normalize
        self.params = params
        if self.params is None:
            self.params = {
                "vector_size": 64,
                "window": 10,
                "min_count": 1,
                "epochs": 20,
                "seed": 42
            }

    def preprocess(self, input_df):
        if self.text_normalize:
            x = self.text_normalize(input_df[self.cols])
        else:
            x = input_df[self.cols].fillna("NaN")
        return x

    def fit(self, input_df):
        text = self.preprocess(input_df)
        corpus = [TaggedDocument(words=x, tags=[i]) for i, x in enumerate(text)]
        self.params["documents"] = corpus
        self.model = Doc2Vec(**self.params, hashfxn=hashfxn)
        result = np.array([self.model.infer_vector(x) for x in text])
        out_df = pd.DataFrame(result)
        return out_df.add_prefix(f"{self.cols}_doc2vec_").add_suffix("_feature")

    def transform(self, input_df):
        text = self.preprocess(input_df)
        result = np.array([self.model.infer_vector(x) for x in text])
        out_df = pd.DataFrame(result)
        return out_df.add_prefix(f"{self.cols}_doc2vec_").add_suffix("_feature")


def hashfxn(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


class BertBlock(AbstractBaseBlock):
    def __init__(self, cols: str, n_components: int = 50, name = "svd", model_name="bert-base-uncased", max_len=128, config=None):
        self.cols = cols
        self.n_components = n_components
        self.name = name
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_len
        self.config = config

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, _ = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()

    def confirm_cumulative_contribution_rate(self, input_df):
        x = np.stack(self.create_text_vector(input_df, task="train"))
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def create_text_vector(self, input_df, task: str = "train"):
        if not os.path.isfile(self.config.output_dir + f"{task}_bert_{self.cols}_feature.pkl"):
            _input_df = input_df.copy()
            _input_df["bert_feature"] = _input_df[self.cols].fillna("NaN").progress_apply(lambda x: self.vectorize(x).reshape(-1))
            _input_df[["bert_feature"]].to_pickle(self.config.output_dir + f"{task}_bert_{self.cols}_feature.pkl")
        else:
            _input_df = pd.read_pickle(self.config.output_dir + f"{task}_bert_{self.cols}_feature.pkl")
        return _input_df["bert_feature"].to_numpy()

    def fit(self, input_df):
        x = np.stack(self.create_text_vector(input_df, task="train"))
        self.decomp.fit(x)
        x = self.decomp.transform(x)
        out_df = pd.DataFrame(x)
        return out_df.add_prefix(f"{self.cols}_bert_").add_suffix(f"_{self.name}_feature")

    def transform(self, input_df):
        x = np.stack(self.create_text_vector(input_df, task="test"))
        x = self.decomp.transform(x)
        out_df = pd.DataFrame(x)
        return out_df.add_prefix(f"{self.cols}_bert_").add_suffix(f"_{self.name}_feature")


class UniversalSentenceEncoderBlock(AbstractBaseBlock):
    def __init__(self, cols: str, n_components: int = 50, name = "svd", url: str = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3", config=None):
        self.cols = cols
        self.n_components = n_components
        self.url = url
        self.config = config
        self.name = name
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)
        self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self.embed = hub.load(url)

    def confirm_cumulative_contribution_rate(self, input_df):
        x = np.stack(self.create_text_vector(input_df, task="train"))
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def create_text_vector(self, input_df, task: str = "train"):
        if not os.path.isfile(self.config.output_dir + f"{task}_universal_{self.cols}_feature.pkl"):
            _input_df = input_df.copy()
            _input_df["universal_feature"] = _input_df[self.cols].fillna("NaN").progress_apply(lambda x: self.embed(x).numpy().reshape(-1))
            _input_df[["universal_feature"]].to_pickle(self.config.output_dir + f"{task}_universal_{self.cols}_feature.pkl")
        else:
            _input_df = pd.read_pickle(self.config.output_dir + f"{task}_universal_{self.cols}_feature.pkl")
        return _input_df["universal_feature"].to_numpy()

    def fit(self, input_df):
        x = np.stack(self.create_text_vector(input_df, task="train"))
        self.decomp.fit(x)
        x = self.decomp.transform(x)
        out_df = pd.DataFrame(x)
        return out_df.add_prefix(f"{self.cols}_universal_").add_suffix(f"_{self.name}_feature")

    def transform(self, input_df):
        x = np.stack(self.create_text_vector(input_df, task="test"))
        x = self.decomp.transform(x)
        out_df = pd.DataFrame(x)
        return out_df.add_prefix(f"{self.cols}_universal_").add_suffix(f"_{self.name}_feature")


class FastTextEmbeddingFeatureBlock(AbstractBaseBlock):
    def __init__(self, cols: str, n_components: int = 50, name = "svd", fast_model=None):
        self.cols = cols
        self.n_components = n_components
        self.fast_model = fast_model
        self.name = name
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)

    def confirm_cumulative_contribution_rate(self, input_df):
        x = input_df[self.cols].progress_apply(lambda x: self.fast_model.get_sentence_vector(x.replace("\n", "")))
        x = np.stack(x.values)
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def fit(self, input_df):
        X = input_df[self.cols].progress_apply(lambda x: self.fast_model.get_sentence_vector(x.replace("\n", "")))
        X = np.stack(X.values)
        self.decomp.fit(X)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_fasttext_embedding_").add_suffix(f"_{self.name}_feature")

    def transform(self, input_df):
        X = input_df[self.cols].progress_apply(lambda x: self.fast_model.get_sentence_vector(x))
        X = np.stack(X.values)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_fasttext_embedding_").add_suffix(f"_{self.name}_feature")


class SimpleTokenizer:
    def tokenize(self, text: str):
        return text.split()


class W2VSWEMBlock(AbstractBaseBlock):
    """
    Simple Word-Embeddingbased Models (SWEM)
    https://arxiv.org/abs/1805.09843v1
    """

    def __init__(self, cols: str, n_components: int = 50, n: int = 2, name = "svd", tokenizer=None, mode: str = "average", model = None, oov_initialize_range=(-0.01, 0.01)):
        self.cols = cols
        self.n_components = n_components
        self.n = n
        self.tokenizer = tokenizer
        self.mode = mode
        self.model = model
        self.vocab = set(self.model.vocab.keys())
        self.embedding_dim = self.model.vector_size
        self.name = name
        self.oov_initialize_range = oov_initialize_range

        if self.oov_initialize_range[0] > self.oov_initialize_range[1]:
            raise ValueError("Specify valid initialize range: "
                             f"[{self.oov_initialize_range[0]}, {self.oov_initialize_range[1]}]")
            
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)

    def confirm_cumulative_contribution_rate(self, input_df):
        x = self.get_swem_vector(input_df)
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def get_word_embeddings(self, text):
        np.random.seed(abs(hash(text)) % (10 ** 8))
        vectors = []
        for word in self.tokenizer.tokenize(text):
            if word in self.vocab:
                vectors.append(self.model[word])
            else:
                vectors.append(np.random.uniform(self.oov_initialize_range[0],
                                                 self.oov_initialize_range[1],
                                                 self.embedding_dim))
        return np.array(vectors)

    def average_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]

    def hierarchical_pooling(self, text, n):
        word_embeddings = self.get_word_embeddings(text)

        text_len = word_embeddings.shape[0]
        if n > text_len:
            raise ValueError(f"window size must be less than text length / window_size:{n} text_length:{text_len}")
        window_average_pooling_vec = [np.mean(word_embeddings[i:i + n], axis=0) for i in range(text_len - n + 1)]

        return np.max(window_average_pooling_vec, axis=0)

    def get_swem_vector(self, input_df):
        if self.mode == "average":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.average_pooling(x)).values)
        elif self.mode == "max":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.max_pooling(x)).values)
        elif self.mode == "concat":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.concat_average_max_pooling(x)).values)
        elif self.mode == "hierachical":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.hierarchical_pooling(x, n=self.n)).values)
        else:
            raise ValueError(f"{self.mode} does not exist")

    def fit(self, input_df):
        X = self.get_swem_vector(input_df)
        self.decomp.fit(X)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_swem_{self.mode}_").add_suffix(f"_{self.name}_feature")

    def transform(self, input_df):
        X = self.get_swem_vector(input_df)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_swem_{self.mode}_").add_suffix(f"_{self.name}_feature")


class FastTextSWEMBlock(AbstractBaseBlock):
    """
    Simple Word-Embeddingbased Models (SWEM)
    https://arxiv.org/abs/1805.09843v1
    """

    def __init__(self, cols: str, n_components: int = 50, n: int = 2, name = "svd", tokenizer=None, mode: str = "average", fast_model = None):
        self.cols = cols
        self.n_components = n_components
        self.n = n
        self.tokenizer = tokenizer
        self.mode = mode
        self.fast_model = fast_model
        self.embedding_dim = self.fast_model.get_dimension()
        self.name = name
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)

    def confirm_cumulative_contribution_rate(self, input_df):
        x = self.get_swem_vector(input_df)
        x = self.decomp.fit_transform(x)
        print(f"Cumulative contribution rate:{np.sum(self.decomp.explained_variance_ratio_)}")

    def get_word_embeddings(self, text):
        np.random.seed(abs(hash(text)) % (10 ** 8))

        vectors = []
        for word in self.tokenizer.tokenize(text):
            vectors.append(self.fast_model.get_word_vector(word))
        return np.array(vectors)

    def average_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]

    def hierarchical_pooling(self, text, n):
        word_embeddings = self.get_word_embeddings(text)

        text_len = word_embeddings.shape[0]
        if n > text_len:
            raise ValueError(f"window size must be less than text length / window_size:{n} text_length:{text_len}")
        window_average_pooling_vec = [np.mean(word_embeddings[i:i + n], axis=0) for i in range(text_len - n + 1)]

        return np.max(window_average_pooling_vec, axis=0)

    def get_swem_vector(self, input_df):
        if self.mode == "average":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.average_pooling(x)).values)
        elif self.mode == "max":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.max_pooling(x)).values)
        elif self.mode == "concat":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.concat_average_max_pooling(x)).values)
        elif self.mode == "hierachical":
            return np.stack(input_df[self.cols].fillna("NaN").str.replace("\n", " ").map(lambda x: self.hierarchical_pooling_pooling(x, n=self.n)).values)
        else:
            raise ValueError(f"{self.mode} does not exist")

    def fit(self, input_df):
        X = self.get_swem_vector(input_df)
        self.decomp.fit(X)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_swem_{self.mode}_").add_suffix(f"_{self.name}_feature")

    def transform(self, input_df):
        X = self.get_swem_vector(input_df)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_swem_{self.mode}_").add_suffix(f"_{self.name}_feature")


class W2VSCDVEmbedderBlock(AbstractBaseBlock):
    def __init__(self, cols, n_components, tokenizer, model, name):
        self.cols = cols
        self.n_components = n_components
        self.tokenizer = tokenizer
        self.model = model
        self.name = name
        self.scdv = SCDVEmbedder(w2v=self.model, tokenizer=self.tokenizer)
        if self.name == "svd" or self.name is None:
            self.name = "svd"
            self.decomp = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.name == "nmf":
            self.decomp = NMF(n_components=self.n_components, random_state=42)
        elif self.name == "pca":
            self.decomp = PCA(n_components=self.n_components, random_state=42)

    def fit(self, input_df):
        self.scdv.fit(input_df[self.cols])
        X = self.scdv.transform(input_df[self.cols])
        self.decomp.fit(X)
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_scdv_").add_suffix(f"_{self.name}_feature")

    def transform(self, input_df):
        X = self.scdv.transform(input_df[self.cols])
        X = self.decomp.transform(X)
        out_df = pd.DataFrame(X)
        return out_df.add_prefix(f"{self.cols}_scdv_").add_suffix(f"_{self.name}_feature")


def tokenizer(x: str):
    return x.split()


class GetLanguageLabelBlock(AbstractBaseBlock):
    """
    言語判定するブロック
    """
    def __init__(self, cols: str, fast_model=None):
        self.cols = cols
        self.fast_model = fast_model

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.cols] = input_df[self.cols].fillna("").map(lambda x: self.fast_model.predict(x.replace("\n", ""))[0][0])
        return out_df.add_prefix("lang_label_")
