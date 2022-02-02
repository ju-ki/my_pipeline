import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from matplotlib_venn import venn2


def create_value_count_plot(input_df: pd.DataFrame, col_list: List[str], n=5):
    """
    指定したカラム(objectやcategory)データの数をプロットする関数

    args:
       input_df : pd.DataFrame
       col_list : list
       n : Argument to specify how many unique values should be plotted (default=5)
    """
    for col in col_list:
        plt.subplots(figsize=(8, 8))
        print(f"{col}: \n {input_df[col].value_counts()[:n]}")
        input_df[col].value_counts().plot.bar()
        plt.show()
        print("***" * 40)


def plot_intersection(left, right, column, set_labels, ax=None):
    left_set = set(left[column])
    right_set = set(right[column])
    venn2(subsets=(left_set, right_set), set_labels=set_labels, ax=ax)
    return ax