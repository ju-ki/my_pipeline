import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMModel
from typing import Optional, Dict
from .base import BaseModel


class MyLGBMModel(BaseModel):

    # ref:https://qiita.com/tubo/items/f83a97f2488cc1f40088 tuboさんのベースラインから
    #     :https://signate.jp/competitions/402/discussions/lgbm-baseline-except-text-vs-include-text-lb07994-1　masatoさんのベースラインから
    def __init__(self, model_params, fit_params: Optional[Dict]):
        self.model_params = model_params
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}

    def build_model(self):
        self.model = LGBMModel(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[[valid_x, valid_y]],
            **self.fit_params
        )
        return self.model

    def predict(self, est, valid_x):
        preds = est.predict(valid_x)
        return preds

    def visualize_importance(self,  train_feat_df: pd.DataFrame):
        feature_importance_df = pd.DataFrame()
        num = 0
        for i, model in self.models.items():
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = train_feat_df.columns
            _df['fold'] = num + 1
            feature_importance_df = pd.concat([feature_importance_df, _df],
                                              axis=0, ignore_index=True)
            num += 1

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
        ax.set_title('Lightgbm Feature Importance')
        ax.grid()
        plt.show()