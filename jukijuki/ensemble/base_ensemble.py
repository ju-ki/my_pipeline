import numpy as np
import pandas as pd
from typing import Union, List


class Ensemble:
    def __init__(self, exp_name=Union[List[str], str], model_name=Union[List[str], str], path=Union[List[str], str], config=None, target=None, metric=None,  is_logarithm=False):
        self.exp_name = exp_name
        self.model_name = model_name
        self.path = path
        self.config = config
        self.target = target
        self.metric = metric
        self.is_logarithm = is_logarithm

    def fit(self):
        self.oof_df = pd.DataFrame()
        self.sub_df = pd.DataFrame()
        if type(self.exp_name) == str:
            self.exp_name = [self.exp_name]
        if type(self.model_name) == str:
            self.model_name = [self.model_name]
        if type(self.path) == str:
            self.path = [self.path]
        for dir in self.path:
            for exp in self.exp_name:
                for model in self.model_name:
                    oof_path = f"{dir}/{exp}/{exp}_{model}_oof.csv"
                    sub_path = f"{dir}/{exp}/{exp}_{model}_sub.csv"
                    _df = pd.read_csv(oof_path).rename(columns={"oof": f"{exp}_{model}"})
                    _df1 = pd.read_csv(sub_path).rename(columns={self.config.target_col: f"{exp}_{model}"})
                    if self.is_logarithm:
                        _df1[f"{exp}_{model}"] = np.log1p(_df1[f"{exp}_{model}"])
                    if self.target is not None:
                        print(f"{exp}_{model}")
                        print(self.metric(self.target.values, _df[f"{exp}_{model}"].values))
                    self.oof_df = pd.concat([self.oof_df, _df], axis=1)
                    self.sub_df = pd.concat([self.sub_df, _df1], axis=1)
            self.sub_df = self.sub_df.loc[:, ~self.sub_df.columns.duplicated()]
            return self.oof_df, self.sub_df

    def average_ensemble(self):
        mean_oof = np.mean(self.oof_df, axis=1)
        print("mean ensemble score")
        print(self.metric(self.target.values, mean_oof.values))
        pred = np.mean(self.sub_df, axis=1)
        if self.is_logarithm:
            pred = np.expm1(pred)
        self.oof_df["oof"] = mean_oof
        self.sub_df[self.config.target_col] = pred
        return self.oof_df[["oof"]], self.sub_df[[self.config.target_col]]

    def median_ensemble(self):
        median_oof = np.median(self.oof_df, axis=1)
        print("median ensemble score")
        print(self.metric(self.target.values, median_oof))
        pred = np.median(self.sub_df, axis=1)
        if self.is_logarithm:
            pred = np.expm1(pred)
        self.oof_df["oof"] = median_oof
        self.sub_df[self.config.target_col] = pred
        return self.oof_df[["oof"]], self.sub_df[[self.config.target_col]]