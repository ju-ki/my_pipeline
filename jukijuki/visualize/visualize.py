import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=25)
    plt.yticks(tick_marks, fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    plt.ylabel('True label', fontsize=25)
    plt.title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontsize=20,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


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
    wordcloud.generate(" ".join(input_df[col].fillna("NaN")))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


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
        unigram_df = get_top_n_words(input_df[col].fillna("NaN"), n=n, stop_words=stop_words)
        plot_freq_word(unigram_df, n=n, name="unigram")
    elif ngram == 2:
        bigram_df = get_top_n_bigram(input_df[col].fillna("NaN"), n=n, stop_words=stop_words)
        plot_freq_word(bigram_df, n=n, name="bigram")
    elif ngram == 3:
        trigram_df = get_top_n_trigram(
            input_df[col].fillna("NaN"), n=n, stop_words=stop_words)
        plot_freq_word(trigram_df, n=n, name="trigram")
    elif ngram == 4:
        print("***" * 40)
        unigram_df = get_top_n_words(input_df[col].fillna("NaN"), n=n, stop_words=stop_words)
        plot_freq_word(unigram_df, n=n, name="unigram")
        print("***" * 40)
        bigram_df = get_top_n_bigram(input_df[col].fillna("NaN"), n=n, stop_words=stop_words)
        plot_freq_word(bigram_df, n=n, name="bigram")
        print("***" * 40)
        trigram_df = get_top_n_trigram(
            input_df[col]fillna("NaN"), n=n, stop_words=stop_words)
        plot_freq_word(trigram_df, n=n, name="trigram")


def plot_intersection(left, right, column, set_labels, ax=None):
    left_set = set(left[column])
    right_set = set(right[column])
    venn2(subsets=(left_set, right_set), set_labels=set_labels, ax=ax)
    return ax


def plot_right_left_intersection(train_df, test_df, columns='__all__'):
    """
        2つのデータフレームのカラムの共通集合を可視化
        Example usage:
           fig, _ = plot_right_left_intersection(train_df, test_df)
           fig.tight_layout()
    """
    if columns == '__all__':
        columns = set(train_df.columns) & set(test_df.columns)

    columns = list(columns)
    nfigs = len(columns)
    ncols = 6
    nrows = - (- nfigs // ncols)
    fig, axes = plt.subplots(figsize=(3 * ncols, 3 * nrows), ncols=ncols, nrows=nrows)
    axes = np.ravel(axes)
    for c, ax in zip(columns, axes):
        plot_intersection(train_df, test_df, column=c, set_labels=('Train', 'Test'), ax=ax)
        ax.set_title(c)
    return fig, ax