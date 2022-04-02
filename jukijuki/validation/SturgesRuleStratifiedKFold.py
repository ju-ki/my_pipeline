import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def struges_skf(X, y, n_splits=5, shuffle=True, random_state=42):
    num_bins = int(np.floor(1 + np.log2(len(X))))
    X["bins"] = pd.cut(y, bins=num_bins, labels=False)
    fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    cv = list(fold.split(X, X["bins"]))
    return cv