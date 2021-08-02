import pandas as pd
import numpy as np
import category_encoders as ce
import xfeat
from util import AbstractBaseBlock


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

    def fit(self, input_df, y=None):
        self.encoder = ce.OrdinalEncoder()
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(
            input_df[self.cols]).add_prefix("LE_")
        return out_df.astype("category")


class CountEncodingBlock(AbstractBaseBlock):
    """CountEncodingを行なう block"""

    def __init__(self, column: str):
        self.column = column

    def fit(self, input_df, y=None):
        vc = input_df[self.column].value_counts()
        # master_df = read_whole_df()
        # vc = master_df[self.column].value_counts()
        self.count_ = vc
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].map(self.count_)
        return out_df.add_prefix('CE_')


class OneHotEncoding(AbstractBaseBlock):
    def __init__(self, column, min_count=30):
        self.column = column
        self.min_count = min_count

    def fit(self, input_df, y=None):
        x = input_df[self.column]
        vc = x.value_counts()
        categories = vc[vc > self.min_count].index
        self.categories_ = categories

        return self.transform(input_df)

    def transform(self, input_df):
        x = input_df[self.column]
        cat = pd.Categorical(x, categories=self.categories_)
        out_df = pd.get_dummies(cat)
        out_df.columns = out_df.columns.tolist()
        return out_df.add_prefix(f'{self.column}=').astype("category")


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

    def __init__(self, group_key, group_values, agg_methods=["max", "min", "mean", "std", "median", max_min, q75_q25]):
        self.group_key = group_key
        self.group_values = group_values
        self.agg_methods = agg_methods

    def fit(self, input_df):
        self.agg_df, self.agg_col = xfeat.aggregation(input_df,
                                                      agg_methods=self.agg_methods,
                                                      group_key=self.group_key,
                                                      group_values=self.group_values)
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.agg_df
        return out_df


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
