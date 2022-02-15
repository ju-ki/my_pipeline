import pandas as pd
import numpy as np
import category_encoders as ce
import xfeat
from typing import List, Union, Optional
from sklearn.model_selection import KFold
from .utils import AbstractBaseBlock


class GetCrossFeatureBlock(AbstractBaseBlock):
    """
    カテゴリ変数同士を組み合わせることができるブロック

    cols:
       exclude_columns:除外したいカラム(ex:id, user_id etc..), デフォルトはNone
       r:いくつ変数を組み合わせるか. デフォルトは2

    """

    def __init__(self, exclude_columns=None, r=2):
        self.exclude_columns = exclude_columns
        self.r = r

    def transform(self, input_df):
        self.encoder = xfeat.Pipeline([
            xfeat.SelectCategorical(exclude_cols=self.exclude_columns),
            xfeat.ConcatCombination(output_suffix="_re", r=self.r)
        ])
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(input_df)
        return out_df


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
    def __init__(self, cols):
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

    def __init__(self, cols: str, is_whole: bool = False):
        self.cols = cols
        self.is_whole = is_whole

    def fit(self, input_df: pd.DataFrame, y=None):
        vc = input_df[self.cols].value_counts()
        if self.is_whole:
            master_df = input_df
            vc = master_df[self.column].value_counts()
        self.count_ = vc
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.cols] = input_df[self.cols].map(self.count_)
        return out_df.add_prefix('CE_')


class OneHotEncoding(AbstractBaseBlock):
    def __init__(self, cols, min_count=30):
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


def max_min(x):
    return x.max() - x.min()


def q75_q25(x):
    return x.quantile(0.75) - x.quantile(0.25)


class AggregationBlock(AbstractBaseBlock):
    """
    集約変数を作成するブロック

    cols:
      group_key:元になる変数
      group_values:集約を行う対象の変数
      agg_methods:どのような方法で集約するか. デフォルトはmax, min, meanの三つ
    """

    def __init__(self, group_key: str, group_values: List[str], agg_methods=None):
        self.group_key = group_key
        self.group_values = group_values
        self.agg_methods = agg_methods
        if self.agg_methods is None:
            self.agg_methods = ["max", "min", "mean"]

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        self.agg_df, self.agg_col = xfeat.aggregation(
            input_df=input_df,
            group_key=self.group_key,
            group_values=self.group_values,
            agg_methods=self.agg_methods
        )

        out_df = self.agg_df[self.agg_col]
        return out_df

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        out_df = self.agg_df
        return out_df


#

class RankingBlock(AbstractBaseBlock):
    def __init__(self, group_key, group_values):
        self.group_key = group_key
        self.group_values = group_values

        self.df = None

    def fit(self, input_df, y=None):
        new_df = []
        new_cols = []

        for col in self.group_values:
            new_cols.append(f"ranking_{col}_grpby_{self.group_key}")
            df__agg = (input_df[[col] + [self.group_key]].groupby(self.group_key)
                       [col].rank(ascending=False, method="min"))
            new_df.append(df__agg)
            self.df = pd.concat(new_df, axis=1)
        self.df.columns = new_cols

    def transform(self, input_df):
        return self.df


class TargetEncodingBlock(AbstractBaseBlock):
    """
       ceを使用したtarget_encoding
    """

    def __init__(self, cols, target):
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
    def __init__(self, cols: str, target: str, n_fold: int = 5, verbosity=True):
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


class GroupingEngine:

    def __init__(self, group_key, group_values, agg_methods):
        self.group_key = group_key
        self.group_values = group_values

        ex_trans_methods = ["val-mean", "z-score"]
        self.ex_trans_methods = [
            m for m in agg_methods if m in ex_trans_methods]
        self.agg_methods = [
            m for m in agg_methods if m not in self.ex_trans_methods]
        self.df = None

    def fit(self, input_df, y=None):
        new_df = []
        for agg_method in self.agg_methods:

            for col in self.group_values:
                if callable(agg_method):
                    agg_method_name = agg_method.__name__
                else:
                    agg_method_name = agg_method

                new_col = f"agg_{agg_method_name}_{col}_grpby_{self.group_key}"
                df_agg = (input_df[[col] + [self.group_key]
                                   ].groupby(self.group_key)[[col]].agg(agg_method))
                df_agg.columns = [new_col]
                new_df.append(df_agg)
        self.df = pd.concat(new_df, axis=1).reset_index()

    def transform(self, input_df):
        output_df = pd.merge(
            input_df[[self.group_key]], self.df, on=self.group_key, how="left")
        if len(self.ex_trans_methods) != 0:
            output_df = self.ex_transform(input_df, output_df)
        output_df.drop(self.group_key, axis=1, inplace=True)
        return output_df

    def ex_transform(self, df1, df2):
        """
        df1: input_df
        df2: output_df
        return: output_df (added ex transformed features)
        """

        if "val-mean" in self.ex_trans_methods:
            df2[self._get_col("val-mean")] = df1[self.group_values].values - \
                df2[self._get_col("mean")].values
        if "z-score" in self.ex_trans_methods:
            df2[self._get_col("z-score")] = (df1[self.group_values].values - df2[self._get_col("mean")].values) \
                / (df2[self._get_col("std")].values + 1e-3)
    #         # df2[self._get_col("z-score")] = (df1[self.group_values] - df1[[key]+self.group_values].groupby(key).transform("mean")) \
    #         #                                                     / (df1[[key]+self.group_values].groupby(key).transform("std") + 1e-8)
        return df2

    def _get_col(self, method):
        return np.sort([f"agg_{method}_{group_val}_grpby_{self.group_key}" for group_val in self.group_values])

    def fit_transform(self, input_df, y=None):
        self.fit(input_df, y=y)
        return self.transform(input_df)


class DiffGroupingEngine(AbstractBaseBlock):
    def __init__(self, group_key, group_values, num_diffs):
        self.group_key = group_key
        self.group_values = group_values
        self.diffs = num_diffs

        self.df = None

    def fit(self, input_df, y=None):
        dfs = []
        for nd in self.diffs:
            _df = input_df.groupby(self.group_key)[self.group_values].diff(nd)
            _df.columns = [
                f'diff={nd}_{col}_grpby_{self.group_key}' for col in self.group_values]
            dfs.append(_df)
        self.df = pd.concat(dfs, axis=1)

    def transform(self, input_df):
        return self.df


class ShiftGroupingEngine(AbstractBaseBlock):
    def __init__(self, group_key, group_values, num_shifts):
        self.group_key = group_key
        self.group_values = group_values
        self.shifts = num_shifts

        self.df = None

    def fit(self, input_df, y=None):
        dfs = []
        for ns in self.shifts:
            _df = input_df.groupby(self.group_key)[self.group_values].shift(ns)
            _df.columns = [
                f'shift={ns}_{col}_grpby_{self.group_key}' for col in self.group_values]
            dfs.append(_df)
        self.df = pd.concat(dfs, axis=1)

    def transform(self, input_df):
        return self.df


class PctChGroupingEngine(AbstractBaseBlock):
    """
    Groupごとの変化分を計算
    """

    def __init__(self, group_key, group_values, num_pctchs):
        self.group_key = group_key
        self.group_values = group_values
        self.num_pctchs = num_pctchs

        self.df = None

    def fit(self, input_df, y=None):
        dfs = []
        for n in self.num_pctchs:
            _df = input_df.groupby(self.group_key)[
                self.group_values].pct_change(n)
            _df.columns = [
                f'pct_change={n}_{col}_grpby_{self.group_key}' for col in self.group_values]
            dfs.append(_df)
        self.df = pd.concat(dfs, axis=1)

    def transform(self, input_df):
        return self.df
