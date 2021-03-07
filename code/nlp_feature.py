import numpy as np
import pandas as pd
import string
import texthero as hero
from nltk.util import ngrams
from fasttext import load_model


class GetCountString(AbstractBaseBlock):
    
    """
    文字数を取得するブロック
    """
  def __init__(self, cols):
    self.cols = cols

  def transform(self, input_df):
    out_df = pd.DataFrame()
    out_df[self.cols] = input_df[self.cols].str.len().fillna("")
    return out_df.add_prefix("count_string_")

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