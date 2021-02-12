import numpy as np
import pandas as pd
import texthero as hero


def remove_exclude_punctuation(input_df, text_col):
    """
    htmlタグと句読点以外を除去.(文単位で扱うことができる)
    """
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.remove_brackets
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

def remove_include_puctuation(input_df, text_col):
    """
    htmlタグや句読点を除去.(単語単位で扱うことができる)
    """
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.remove_brackets,
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

def get_html_tags_only(input_df, text_col):
    htmls = input_df[text_col]
    html_tags = []
    for html in htmls:
        tmp = re.sub(r"\s", "", html)
        tmp = re.sub(r"\d", "", tmp)
        tmp = " ".join(re.findall(r"(?<=<).*?(?=>)", tmp))
        html_tags.append(tmp)

    return pd.Series(html_tags)