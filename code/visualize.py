import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
%matplotlib inline
sns.set()

def visualize_binary_data(input_df, cols, target_cols, pos_var, neg_var):
    """
    二値分類用の割合データ
    
    cols:
      pos_var:問題設定となっている方のターゲット
      neg_var:pos_varとは違うターゲット
    """
    cross_table = pd.crosstab(input_df[cols], input_df[target_cols], margins=True)
    pos_name = cross_table[pos_var] / cross_table["All"]
    neg_name = cross_table[neg_var] / cross_table["All"]
    
    cross_table["pos_var"] = pos_name
    cross_table["neg_var"] = neg_name
    
    cross_table = cross_table.drop(index=["All"])
    display(cross_table)
    
    tmp = cross_table[["pos_var", "neg_var"]]
    tmp = tmp.sort_values(by="pos_var", ascending=False)
    tmp.plot.bar(stacked=True, title='pos_var vs neg bar by choice column.')
    plt.xlabel(cols)
    plt.ylabel("Percentage")
    plt.show()

def show_scatterplot(input_df, x, y, hue=None, reg=True, title=None, xlabel=None, ylabel=None):
    """
    散布図
    """
    plt.figure(figsize=(8, 6))
    if hue is not None:
        input_df = input_df.sort_values(hue)
    if reg:
        sns.regplot(x=x, y=y, data=input_df, scatter=False, color="red")
    sns.scatterplot(data=input_df, x=x, y=y, hue=hue, s=200, palette="Set1", alpha=0.5)
    if title is not None:
        plt.title(None)
        
    if xlabel is not None:
        plt.xlabel(xlabel)
        
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()