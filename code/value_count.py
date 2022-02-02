import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


def plot_value_count(df: pd.DataFrame, col_name: str, figsize=(20, 7), display_name=None, params=None, is_plotly=True):
    x = df[col_name].value_counts().index
    y = df[col_name].value_counts().values
    if display_name is None:
        display_name = col_name.capitalize()
    if is_plotly:
        if params is None:
            params = {
                "texttemplate": "%{text:.2s}",
                "marker_color": "darkblue",
                "marker_line_color": "white",
                "textfont": dict(size=18),
                "textposition": 'inside',
                "opacity": 0.6
            }

        fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition="outside")])
        fig.update_traces(params)
        fig.update_xaxes(title_text=display_name, nticks=20, titlefont=dict(size=18), tickfont=dict(size=16))
        fig.update_yaxes(title_text="Value Counts", titlefont=dict(size=18), tickfont=dict(size=16))
        return fig.show()
    else:
        plt.style.use("seaborn-dark")
        if params is None:
            params = {
                "xtick.labelsize": 14,
                "axes.labelsize": 18
            }

        plt.rcParams.update(params)
        fig, ax = plt.subplots(figsize=figsize)
        plot1 = sns.barplot(x=x, y=y)
        plot1.set_xlabel(display_name)
        plot1.set_ylabel("Value Counts")
        fig.show()