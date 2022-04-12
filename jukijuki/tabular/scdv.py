import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.mixture import GaussianMixture


class SCDVEmbedder(TransformerMixin, BaseEstimator):
    def __init__(self, w2v, tokenizer, k=5):
        self.w2v = w2v
        self.vocab = set(self.w2v.vocab.keys())
        self.tokenizer = tokenizer
        self.k = k
        self.topic_vector = None
        self.tv = TfidfVectorizer(tokenizer=self.tokenizer)

    def __assert_if_not_fitted(self):
        assert self.topic_vector is not None, \
            "SCDV model has not been fitted"

    def __create_topic_vector(self, corpus: pd.Series):
        self.tv.fit(corpus)
        self.doc_vocab = set(self.tv.vocabulary_.keys())

        self.use_words = list(self.vocab & self.doc_vocab)
        self.use_word_vectors = np.array([
            self.w2v[word] for word in self.use_words])
        w2v_dim = self.use_word_vectors.shape[1]
        self.clf = GaussianMixture(
            n_components=self.k,
            random_state=42,
            covariance_type="tied")
        self.clf.fit(self.use_word_vectors)
        word_probs = self.clf.predict_proba(
            self.use_word_vectors)
        world_cluster_vector = self.use_word_vectors[:, None, :] * word_probs[
            :, :, None]

        doc_vocab_list = list(self.tv.vocabulary_.keys())
        use_words_idx = [doc_vocab_list.index(w) for w in self.use_words]
        idf = self.tv.idf_[use_words_idx]
        topic_vector = world_cluster_vector.reshape(-1, self.k * w2v_dim) * idf[:, None]
        topic_vector = np.nan_to_num(topic_vector)

        self.topic_vector = topic_vector
        self.vocabulary_ = set(self.use_words)
        self.ndim = self.k * w2v_dim

    def fit(self, X, y=None):
        self.__create_topic_vector(X)

    def transform(self, X):
        tokenized = X.fillna("").map(lambda x: self.tokenizer(x))

        def get_sentence_vector(x: list):
            embeddings = [
                self.topic_vector[self.use_words.index(word)]
                if word in self.vocabulary_
                else np.zeros(self.ndim, dtype=np.float32)
                for word in x
            ]
            if len(embeddings) == 0:
                return np.zeros(self.ndim, dtype=np.float32)
            else:
                return np.mean(embeddings, axis=0)
        return np.stack(
            tokenized.map(lambda x: get_sentence_vector(x)).values
        )