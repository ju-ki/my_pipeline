from catboost import Pool, CatBoost
from typing import List, Optional, Union, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .base import BaseModel


class MyCatModel(BaseModel):
    def __init__(self, model_params, fit_params: Optional[Dict], categorical_features: Optional[Union[List[str], List[int]]]):
        self.model_params = model_params
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}
        self.categorical_features = categorical_features

    def build_model(self):
        self.model = CatBoost(self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_pool = Pool(train_x, train_y, cat_features=self.categorical_features)
        valid_pool = Pool(valid_x, valid_y, cat_features=self.categorical_features)
        self.model = self.build_model()
        self.model.fit(train_pool,
                       plot=False,
                       use_best_model=True,
                       eval_set=[valid_pool],
                       **self.fit_params
                       )
        return self.model

    def predict(self, est, valid_x):
        return est.predict(valid_x)

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
        ax.set_title('Catboost Feature Importance')
        ax.grid()
        plt.show()
