import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplot(df: pd.DataFrame, col_name: str, figsize=(20, 7), display_name=None, params=None):
    plt.style.use("seaborn-dark")
    if params is None:
        params = {
            "xtick.labelsize": 14,
            "axes.labelsize": 18
        }
    if display_name is None:
        display_name = col_name.capitalize()

    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=figsize)
    plot1 = sns.boxplot(df[col_name])
    plot1.set_xlabel(display_name)
    fig.show()