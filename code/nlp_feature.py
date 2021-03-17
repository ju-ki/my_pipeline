import numpy as np
import pandas as pd
import string
import texthero as hero
from nltk.util import ngrams
from fasttext import load_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import Pipeline


class StringLengthBlock(AbstractBaseBlock):
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].str.len()
        return out_df.add_prefix('StringLength_')

class GetCountWord(AbstractBaseBlock):
    
    """単語数を取得するブロック"""
  def __init__(self, cols):
    self.cols = cols

  def transform(self, input_df):
    out_df = pd.DataFrame()
    out_df[self.cols] = cleansing_hero_text(input_df[self.cols])
    out_df[self.cols] = out_df[self.cols].apply(lambda x: len(x.split())).fillna("")
    return out_df.add_prefix("count_word_")


def cleansing_hero_text(text_col):
    """
    Stopwords抜きの関数
    """
  custom_pipeline = [
                   hero.preprocessing.remove_whitespace,
                   hero.preprocessing.drop_no_content,
                   hero.preprocessing.fillna,
                   hero.preprocessing.remove_diacritics,
                   hero.preprocessing.remove_digits,
                   hero.preprocessing.remove_html_tags,
                   hero.preprocessing.remove_urls,
                   hero.preprocessing.remove_brackets,
                   hero.preprocessing.remove_punctuation,
                   hero.preprocessing.stem]
  texts = hero.clean(text_col, custom_pipeline)
  return texts

def cleansing_hero_text_and_stopwords(text_col):
    
    """
    stopwordsを含む
    """
  custom_pipeline = [
                   hero.preprocessing.remove_whitespace,
                   hero.preprocessing.drop_no_content,
                   hero.preprocessing.fillna,
                   hero.preprocessing.remove_diacritics,
                   hero.preprocessing.remove_digits,
                   hero.preprocessing.remove_html_tags,
                   hero.preprocessing.remove_urls,
                   hero.preprocessing.remove_brackets,
                   hero.preprocessing.remove_punctuation,
                   hero.preprocessing.stem,
                   hero.preprocessing.remove_stopwords]
  texts = hero.clean(text_col, custom_pipeline)
  return texts

def line_ngram(line, n=2):
  words = [w for w in line.split(' ') if len(w) != 0]
  return list(ngrams(words, n))

def create_n_gram(x, n=3):
  x = cleansing_hero_text(x)
  x = pd.Series(x).map(lambda r: line_ngram(r, n=n))
  return x

class NameNGramBlock(AbstractBaseBlock):
    def __init__(self, whole_df, col, n=3):
        self.whole_df = whole_df
        self.col = col
        self.n = n

    def fit(self, input_df, y=None):
        name_grams = create_n_gram(self.whole_df[self.col], n=self.n)
        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(20).index

        self.top_grams_ = top_grams
        return self.transform(input_df)

    def transform(self, input_df):
        name_grams = create_n_gram(input_df[self.col], n=self.n)
        output_df = pd.DataFrame()

        for top in self.top_grams_:
            s_top = '-'.join(top)
            output_df[f'{s_top}'] = name_grams.map(lambda x: top in x).map(int)

        return output_df.add_prefix('Name_has_').add_suffix(f'_n={self.n}')

def text_normalization(text):

    # 英語とオランダ語を stopword として指定
    #複数の言語を処理したい場合は逐一変更
    custom_stopwords = nltk.corpus.stopwords.words('english')

    x = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])

    return x
      
class TfidfBlock(AbstractBaseBlock):
    """tfidf x SVD による圧縮を行なう block"""
    def __init__(self, column: str):
        """
        args:
            column: str
                変換対象のカラム名
        """
        self.column = column

    def preprocess(self, input_df):
        x = text_normalization(input_df[self.column])
        return x

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す. 
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = self.preprocess(input_df)
        self.pileline_ = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('svd', TruncatedSVD(n_components=50)),
        ])

        self.pileline_.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pileline_.transform(text)

        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.column}_tfidf_')
      
class TextVectorizer(AbstractBaseBlock):

    def __init__(self,
                 text_columns,
                 cleansing_hero=None,
                 vectorizer=CountVectorizer(),
                 transformer=TruncatedSVD(n_components=128),
                 transformer2=None,
                 name='',
                 ):
        self.text_columns = text_columns
        self.n_components = transformer.n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.transformer2 = transformer2
        self.name = name
        self.cleansing_hero = cleansing_hero

        self.df = None

    def fit(self, input_df, y=None):
        output_df = pd.DataFrame()
        output_df[self.text_columns] = input_df[self.text_columns].astype(str).fillna('missing')
        features = []
        for c in self.text_columns:
            if self.cleansing_hero is not None:
                output_df[c] = self.cleansing_hero(output_df, c)

            sentence = self.vectorizer.fit_transform(output_df[c])
            feature = self.transformer.fit_transform(sentence)

            if self.transformer2 is not None:
                feature = self.transformer2.fit_transform(feature)

            num_p = feature.shape[1]
            feature = pd.DataFrame(feature, columns=[f"{c}_{self.name}{num_p}" + f'={i:03}' for i in range(num_p)])
            features.append(feature)
        output_df = pd.concat(features, axis=1)
        self.df = output_df

    def transform(self, input_df):
        return self.df


class Doc2VecFeatureTransformer(AbstractBaseBlock):

    def __init__(self, text_columns, cleansing_hero=None, params=None, name='doc2vec'):
        self.text_columns = text_columns
        self.cleansing_hero = cleansing_hero
        self.name = name
        self.params = params
        self.df = None

    def fit(self, input_df, y=None):
        dfs = []
        for c in self.text_columns:
            texts = input_df[c].astype(str)
            if self.cleansing_hero is not None:
                texts = self.cleansing_hero(input_df, c)
            texts = [text.split() for text in texts]

            corpus = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(texts)]
            self.params["documents"] = corpus
            model = Doc2Vec(**self.params, hashfxn=hashfxn)

            result = np.array([model.infer_vector(text) for text in texts])
            output_df = pd.DataFrame(result)
            output_df.columns = [f'{c}_{self.name}:{i:03}' for i in range(result.shape[1])]
            dfs.append(output_df)
        output_df = pd.concat(dfs, axis=1)
        self.df = output_df

    def transform(self, dataframe):
        return self.df
    
class GetLanguageLabel(AbstractBaseBlock):
    """
    言語判定するブロック
    """
  def __init__(self, cols):
    self.cols = cols

  def fit(self, input_df):
    self.model = load_model("/content/drive/MyDrive/atmacup10/data/external/lid.176.bin")
    return self.transform(input_df)

  def transform(self, input_df):
    out_df = pd.DataFrame()
    out_df[self.cols] = input_df[self.cols].fillna("").map(lambda x: self.model.predict(x.replace("\n", ""))[0][0])
    return out_df.add_prefix("lang_label_")