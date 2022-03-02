import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
sns.set()


def create_top_ngram_word_plot(input_df: pd.DataFrame, col: str, n=20, stop_words=None, ngram=4):
    """[summary]

    Args:
        input_df (pd.DataFrame): target data
        col (string): string column
        n (int, optional): Argument to specify how many data should be plotted. Defaults to 20.
        stop_words (list, optional): remove words. Defaults to None.
        ngram (int, n=1:unigram, n=2:bigram, n=3:trigram, n=4:all ngram(uni, bi, tri)): Specify which ngrams to plot. Defaults to 4.
    """
    def get_top_n_words(corpus, n=None, stop_words=None):
        vec = CountVectorizer(stop_words=stop_words).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def get_top_n_bigram(corpus, n=None, stop_words=None):
        vec = CountVectorizer(ngram_range=(
            2, 2), stop_words=stop_words).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def get_top_n_trigram(corpus, n=None, stop_words=None):
        vec = CountVectorizer(ngram_range=(
            3, 3), stop_words=stop_words).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_freq_word(df, n=None, name=None):
        plt.figure(figsize=(10, 10))
        ax = sns.barplot(x="freq", y="word", data=pd.DataFrame(
            df, columns=["word", "freq"]))
        plt.title(f"Top {n} {name}")
        plt.xticks(rotation=45)
        plt.xlabel("Frequency")
        plt.ylabel("")
        plt.show()

    if ngram == 1:
        unigram_df = get_top_n_words(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(unigram_df, n=n, name="unigram")
    elif ngram == 2:
        bigram_df = get_top_n_bigram(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(bigram_df, n=n, name="bigram")
    elif ngram == 3:
        trigram_df = get_top_n_trigram(
            input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(trigram_df, n=n, name="trigram")
    elif ngram == 4:
        print("***" * 40)
        unigram_df = get_top_n_words(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(unigram_df, n=n, name="unigram")
        print("***" * 40)
        bigram_df = get_top_n_bigram(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(bigram_df, n=n, name="bigram")
        print("***" * 40)
        trigram_df = get_top_n_trigram(
            input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(trigram_df, n=n, name="trigram")