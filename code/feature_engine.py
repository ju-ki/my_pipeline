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
    out_df = self.encoder.fit_transform(input_df[self.cols]).add_prefix("LE_")
    return out_df
    
class CountEncodingBlock(AbstractBaseBlock):
    
    """
    CountEncodingを行うブロック
    """
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, input_df, y=None):
        self.encoder = ce.CountEncoder()
        return self.transform(input_df)
    
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(input_df[self.cols]).add_prefix("CE_")
        return out_df
    
    
class OneHotEncodingBlock(AbstractBaseBlock):
    """
    OneHotEncodingを行うブロック
    """
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, input_df, y=None):
        self.encoder = ce.OneHotEncoder(use_cat_names=True)
        return self.transform(input_df)
    
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(input_df[self.cols]).add_prefix("OHE_")
        return out_df
    
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
            df__agg = (input_df[[col] + [self.group_key]].groupby(self.group_key)[col].rank(ascending=False, method="min"))
            new_df.append(df__agg)
            self.df = pd.concat(new_df, axis=1)
        self.df.columns = new_cols
        
    def transform(self, input_df):
        return self.df
    
class SimpleTargetEncodingBlock(AbstractBaseBlock):
    """
    シンプルなターゲットエンコーディングを行うブロック
    
    cols:
      cols:ターゲットエンコーディングを行いたいカラム
      target_cols:ターゲット
      smoothing: float
    """
    def __init__(self, cols, target_cols, smoothing=1.0):
        self.cols = cols
        self.target_cols = target_cols
        self.smoothing = smoothing
        
    def fit(self, input_df):
        self.encoder = ce.TargetEncoder(smoothing=self.smoothing)
        return self.transform(input_df)
    
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = self.encoder.fit_transform(input_df[self.cols], input_df[self.target_cols]).add_prefix("TE_")
        return out_df