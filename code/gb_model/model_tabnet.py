import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Optional, Dict
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from .base import BaseModel
from utils.util import decorate


class MyTabNetModel(BaseModel):
    def __init__(self, model_params, fit_params: Optional[Dict]):
        self.model_params = model_params
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}

    def build_model(self):
        self.model = TabNetRegressor(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        train_x, valid_x = train_x.values, valid_x.values
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_name=["train", "valid"],
            **self.fit_params
        )
        return self.model

    def predict(self, est, valid_x):
        valid_x = valid_x.values
        preds = est.predict(valid_x)
        return preds


class MyTabNeClassifierModel(BaseModel):
    def __init__(self, model_params, fit_params):
        self.model_params = model_params
        self.fit_params = fit_params

    def build_model(self):
        self.model = TabNetClassifier(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        train_x = train_x.values
        valid_x = valid_x.values
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_name=["train", "valid"],
            **self.fit_params
        )
        return self.model

    def predict(self, est, valid_x):
        valid_x = valid_x.values
        preds = est.predict_proba(valid_x)[:, 1]
        return preds

    def get_feature_importance(self, train_x: pd.DataFrame):
        feature_importance_df = pd.DataFrame()
        num = 0
        for i, model in self.models.items():
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = train_x.columns
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
        ax.set_title('Tabnet Feature Importance')
        ax.grid()
        plt.show()


class MyPretrainTabnet:
    def __init__(self, model_params, fit_params):
        self.model_params = model_params
        self.fit_params = fit_params
        self.model = None

    def fit(self, train_x, valid_x):
        self.model = TabNetPretrainer(**self.model_params)
        self.model.fit(
            train_x,
            eval_set=[valid_x],
            eval_name=["train"],
            **self.fit_params
        )
        return self.model

    def pretrain_run(self, name: str, train_x: pd.DataFrame, train_y: np.ndarray, cv, output_dir: str = "./"):
        va_idxes = []
        for cv_num, (trn_idx, val_idx) in enumerate(cv):
            print(decorate("start pretraining"))
            tr_x, va_x = train_x.values[trn_idx], train_x.values[val_idx]
            # tr_y, va_y = train_y[trn_idx], train_y[val_idx]
            va_idxes.append(val_idx)
            model = self.fit(tr_x, va_x)
            model_name = f"{name}_FOLD{cv_num}_pretrain_model"
            model.save_model(output_dir + model_name)