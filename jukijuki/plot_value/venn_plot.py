import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from matplotlib_venn import venn2

def plot_intersection(left: pd.DataFrame, right: pd.DataFrame, target_column: str, set_labels: List[str] = None, ax=None):
    """
    Paramters
    ----------
    left: pd.DataFrame
    right: pd.DataFrame
    column:
    
    ### Example use:
        target_columns = train_df.columns.tolist()
        n_cols = 5(example num)
        n_rows = - (- len(target_columns) // n_cols)
        fig, axes = plt.subplots(figsize=(4 * n_cols, 3 * n_rows), ncols=n_cols, nrows=n_rows)
        
        for c, ax in zip(target_columns, np.ravel(axes)):
            plot_intersection(train_df, test_df, target_column=c, ax=ax)
    """
    left_set = set(left[target_column])
    right_set = set(right[target_column])
    venn2(subsets=(left_set, right_set), set_labels=set_labels, ax=ax)
    return ax