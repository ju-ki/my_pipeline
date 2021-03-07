import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
sns.set()

def get_grpby_info(data, group_key, target_col, display_num=None, ascending=False):

  """
  集約した情報を渡す関数
      data: data,
      group_key: 集約したいカラム,
      target_col: 対象となるカラム,
      display_num: 表示する個数（defaultはNone）
      ascending:　defaultはFalse
  """
  print("-------------------------------count------------------------------------")
  display(data.groupby(group_key).agg({target_col:"count"}).sort_values(target_col, ascending=ascending).head(display_num))
  print("-------------------------------mean------------------------------------")
  display(data.groupby(group_key).agg({target_col:"mean"}).sort_values(target_col, ascending=ascending).head(display_num))
  print("-------------------------------median------------------------------------")
  display(data.groupby(group_key).agg({target_col:"mean"}).sort_values(target_col, ascending=ascending).head(display_num))
  print("-------------------------------max------------------------------------")
  display(data.groupby(group_key).agg({target_col:"mean"}).sort_values(target_col, ascending=ascending).head(display_num))
  print("-------------------------------min------------------------------------")
  display(data.groupby(group_key).agg({target_col:"mean"}).sort_values(target_col, ascending=ascending).head(display_num))
  print("-------------------------------std------------------------------------")
  display(data.groupby(group_key).agg({target_col:"mean"}).sort_values(target_col, ascending=ascending).head(display_num))
  
  
def plot_intersection(left, right, column, set_labels, ax=None):
    left_set = set(left[column])
    right_set = set(right[column])
    venn2(subsets=(left_set, right_set), set_labels=set_labels, ax=ax)
    return ax

def plot_right_left_intersection(train_df, test_df, columns='__all__'):
    """2つのデータフレームのカラムの共通集合を可視化"""
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
    
def visualize_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.

    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x='feature_importance', 
                  y='column', 
                  order=order, 
                  ax=ax, 
                  palette='viridis', 
                  orient='h')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Importance')
    ax.grid()
    return fig, ax