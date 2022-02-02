import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_wordcloud(input_df: pd.DataFrame, col: str, stopwords=None):
    """[summary]
    plot word frequency with WordCloud
    Args:
        input_df (pd.DataFrame): target data
        col (string): string column
        stopwords (list, optional): remove words . Defaults to None.
    """
    print(col)
    plt.subplots(figsize=(15, 15))
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color="White", width=1800, height=1000)
    wordcloud.generate(" ".join(input_df[col]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()