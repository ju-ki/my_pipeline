import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from xgboost import XGBModel
from .base import BaseModel


class MyXGBModel(BaseModel):
    """
    Parameters
    --------------
    ref : https://xgboost.readthedocs.io/en/stable/parameter.html
    model_params:
       objective: reg:squarederror reg:logistic binary:logistic multi:softmax
       max_depth: default=6
       learning_rate: default=0.3
       booster: defualt=gbtree, alternative=> gblinear or dart.
    
    fit_params:
       eval_metric: rmse mae logloss mlogloss auc
       early_stopping_rounds: default=None
       verbose: default=1
    """
    def __init__(self, model_params, fit_params=Optional[Dict]):
        self.model_params = model_params
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}

    def build_model(self):
        model = XGBModel(**self.model_params)
        return model

    def fit(self, train_x, train_y, valid_x, valid_y):
        self.model = self.build_model()
        self.model.fit(train_x, train_y,
                       eval_set=[(valid_x, valid_y)],
                       **self.fit_params
                       )
        return self.model

    def predict(self, est, valid_x):
        return est.predict(valid_x)

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
        ax.set_title('XGBoost Feature Importance')
        ax.grid()
        plt.show()
