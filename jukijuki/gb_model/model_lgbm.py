import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMModel
from typing import Optional, Dict, Union, List
from .base import BaseModel


class MyLGBMModel(BaseModel):
    """
    Parameters
    -----------
    ref: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    model_params:
        objective: default=regression mae rmse poison binary multiclass cross_entropy
        boosting: default=gbdt gbdt rf dart goss
        learning_rate: default=0.1
        num_leaves: default=31
        num_threads: default=0
        num_class:default=1
        max_depth:default=-1
        bagging_fraction:default=1.0
        feature_fraction:default=1.0 alias=>colsample_bytree
        lambda_l1:default=0.0
        lambda_l2:default=0.0
        is_unbalance:default=False
    fit_params:
        verbose: default=1
        early_stopping_rounds:None
        eval_metric:mae mse rmse poison auc average_precision binary_logloss binary_error multi_logloss cross_entropy
    """

    # ref:https://qiita.com/tubo/items/f83a97f2488cc1f40088 tuboさんのベースラインから
    #     :https://signate.jp/competitions/402/discussions/lgbm-baseline-except-text-vs-include-text-lb07994-1　masatoさんのベースラインから
    def __init__(self, model_params, fit_params: Optional[Dict], categorical_features: Optional[Union[List[str], List[int]]]):
        self.model_params = model_params
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}
        self.categorical_features = categorical_features

    def build_model(self):
        self.model = LGBMModel(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[[valid_x, valid_y]],
            categorical_feature=self.categorical_features,
            **self.fit_params
        )
        return self.model

    def predict(self, est, valid_x):
        preds = est.predict(valid_x)
        return preds

    def get_feature_importance(self,  train_feat_df: pd.DataFrame):
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