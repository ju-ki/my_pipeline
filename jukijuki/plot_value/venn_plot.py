import numpy as np
import matplotlib.pyplot as plt
from typing import List
from matplotlib_venn import venn2


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