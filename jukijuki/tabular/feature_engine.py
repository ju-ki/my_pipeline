import pandas as pd
import numpy as np
import category_encoders as ce
import xfeat
from typing import List, Union, Optional
from itertools import combinations
from sklearn.model_selection import KFold
from .utils import AbstractBaseBlock


class CrossCategoricalFeatureBlock(AbstractBaseBlock):
    def __init__(self, use_cols: List[str], r: int = 2, fillna: str = "NaN"):
        self.use_cols = use_cols
        self.r = r
        self.fillna = fillna

    def concat_categorical_columns(self, input_df):
        # refs: https://github.com/pfnet-research/xfeat/blob/master/xfeat/cat_encoder/_concat_combination.py
        _df = input_df.copy()
        cols = []
        for cols_pairs in combinations(self.use_cols, r=self.r):
            pairs_cols_str = "&".join(cols_pairs)
            new_col = (pairs_cols_str)
            cols.append(new_col)
            concat_cols = self.use_cols
            new_ser = None
            for col in concat_cols:
                if new_ser is None:
                    new_ser = _df[col].fillna(self.fillna).copy()
                else:
                    new_ser = new_ser + "&" + _df[col].fillna(self.fillna)
            _df[new_col] = new_ser
        return _df[cols]

    def transform(self, input_df):
        return self.concat_categorical_columns(input_df)


class GetAddNumFeatureBlock(AbstractBaseBlock):
    """
    数値変数同士を組み合わせるブロック

    cols:
      exclude_columns:除外したいカラム(ex:id, 数値型のターゲット etc), デフォルトはNone
      r:いくつの変数を組み合わせるか. デフォルトは2
      operator:どのような演算を行うか. デフォルトは加算(+)
    """

    def __init__(self, exclude_columns=None, r=2, operator="+"):
        self.exclude_columns = exclude_columns
        self.operator = operator
        self.r = r

    def transform(self, input_df):
        self.encoder = xfeat.Pipeline([
            xfeat.SelectNumerical(),
            xfeat.ArithmeticCombinations(exclude_cols=self.exclude_columns,
                                         operator=self.operator,
                                         output_suffix="_combi",
                                         r=self.r
                                         )])
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(input_df)
        return out_df


class LabelEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols: str):
        self.cols = cols

    def fit(self, input_df: pd.DataFrame, y=None):
        self.encoder = ce.OrdinalEncoder()
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(
            input_df[self.cols]).add_prefix("LE_")
        return out_df.astype("category")


class CountEncodingBlock(AbstractBaseBlock):
    """CountEncodingを行なう block"""

    def __init__(self, cols: str, is_whole: bool = False, master_df: pd.DataFrame = None):
        self.cols = cols
        self.is_whole = is_whole
        if self.is_whole:
            self.master_df = master_df

    def fit(self, input_df: pd.DataFrame, y=None):
        vc = input_df[self.cols].value_counts()
        if self.is_whole:
            vc = self.master_df[self.column].value_counts()
        self.count_ = vc
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.cols] = input_df[self.cols].map(self.count_)
        return out_df.add_prefix('CE_')


class OneHotEncoding(AbstractBaseBlock):
    def __init__(self, cols: str, min_count: int = 30):
        self.cols = cols
        self.min_count = min_count

    def fit(self, input_df, y=None):
        x = input_df[self.cols]
        vc = x.value_counts()
        categories = vc[vc > self.min_count].index
        self.categories_ = categories

        return self.transform(input_df)

    def transform(self, input_df):
        x = input_df[self.cols]
        cat = pd.Categorical(x, categories=self.categories_)
        out_df = pd.get_dummies(cat)
        out_df.columns = out_df.columns.tolist()
        return out_df.add_prefix(f'{self.cols}=').astype("category")


class MaxMin:
    def __call__(self, x):
        return max(x)-min(x)

    def __str__(self):
        return "max_min"


class Quantile:
    def __call__(self, x):
        return x.quantile(0.75) - x.quantile(0.25)

    def __str__(self):
        return "interquartile_range"


class AggregationBlock(AbstractBaseBlock):
    def __init__(self, key: str, values: List[str], agg_methods: List[str]):
        self.key = key
        self.values = values
        self.agg_methods = agg_methods
        if self.agg_methods is None:
            self.agg_methods = ["max", "min", "mean", "std"]
        ex_trans_methods = ["val-mean", "z-score"]
        self.ex_trans_methods = [
            m for m in self.agg_methods if m in ex_trans_methods]

        self.agg_methods = [
            m for m in self.agg_methods if m not in self.ex_trans_methods]

    def ex_transform(self, df1, df2):
        """
        df1: input_df
        df2: output_df
        return: output_df (added ex transformed features)
        """
        if "val-mean" in self.ex_trans_methods:
            _agg_df, _agg_list = xfeat.aggregation(df1, group_key=self.key, group_values=self.values, agg_methods=["mean"])
            mean_list = [m for m in _agg_list if "mean" in m]
            df2[self._get_col("val-mean")] = df1[self.values].values - _agg_df[mean_list].values

        if "z-score" in self.ex_trans_methods:
            _agg_df, _agg_list = xfeat.aggregation(df1, group_key=self.key, group_values=self.values, agg_methods=["mean", "std"])
            mean_list = [m for m in _agg_list if "mean" in m]
            std_list = [m for m in _agg_list if "std" in m]
            df2[self._get_col("z-score")] = ((df1[self.values].values - _agg_df[mean_list].values)
                                             / (_agg_df[std_list].values + 1e-8))

        return df2

    def _get_col(self, method):
        return [f"agg_{method}_{group_val}_grpby_{self.key}" for group_val in self.values]

    def fit(self, input_df: pd.DataFrame):
        agg_df, agg_list = xfeat.aggregation(input_df=input_df, group_key=self.key, group_values=self.values, agg_methods=self.agg_methods)
        new_col = [self.key] + agg_list
        self.meta_df = agg_df[new_col].drop_duplicates().dropna().reset_index(drop=True)
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.merge(input_df[self.key], self.meta_df, on=self.key, how="left")
        if len(self.ex_trans_methods) != 0:
            out_df = self.ex_transform(input_df, out_df)
        out_df.drop(self.key, axis=1, inplace=True)

        return out_df


class TargetEncodingBlock(AbstractBaseBlock):
    """
       ceを使用したtarget_encoding
    """

    def __init__(self, cols: str, target: str):
        self.cols = cols
        self.target = target
        self.encoder = ce.TargetEncoder(cols=self.cols)

    def fit(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(
            input_df[self.cols], input_df[self.target])
        return out_df.add_prefix("TE_")

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.transform(input_df[self.cols])
        return out_df.add_prefix("TE_")


class KFoldTargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols: str, target: str, n_fold: int = 5, verbosity: bool = True):
        """_summary_
        Refs:
            https://www.guruguru.science/competitions/16/discussions/9e36d002-3ac2-4a96-8336-d407cad7a720/
            https://qiita.com/ground0state/items/6778dc2132abdabaf914
        Args:
            cols (str): col_name
            target (str): target feature
            n_fold (int, optional): num of fold Defaults to 5.
            verbosity (bool, optional): whether display correlation between new feature and target. Defaults to True.
        """
        self.cols = cols
        self.target = target
        self.n_fold = n_fold
        self.verbosity = verbosity

    def create_mapping(self, input_df, y=None):
        self.mapping_df = {}
        self.mean_of_target = input_df[self.target].mean()
        X = input_df[self.cols]
        y = input_df[self.target]
        out_df = pd.DataFrame()
        oof = np.zeros_like(X, dtype=np.float)
        kfold = KFold(n_splits=self.n_fold, shuffle=False)
        for trn_idx, val_idx in kfold.split(X):
            _df = y[trn_idx].groupby(X[trn_idx]).mean()
            _df = _df.reindex(X.unique())
            _df = _df.fillna(_df.mean())
            oof[val_idx] = input_df[self.cols][val_idx].map(_df.to_dict())
        out_df[self.cols] = oof
        self.mapping_df[self.cols] = y.groupby(X).mean()
        if self.verbosity:
            print(f"Correlation between KFold_TE_{self.cols} and {self.target} is {np.corrcoef(y, oof)[0][1] : .5f}")
        return out_df

    def fit(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        out_df = self.create_mapping(input_df)
        return out_df.add_prefix("KFold_TE_")

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        out_df[self.cols] = input_df[self.cols].map(self.mapping_df[self.cols]).fillna(self.mean_of_target)
        return out_df.add_prefix("KFold_TE_")
