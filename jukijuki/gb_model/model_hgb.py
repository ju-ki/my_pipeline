from typing import Optional, Dict
from sklearn.ensemble import HistGradientBoostingClassifier
from .base import BaseModel


class MyHGBClassifierModel(BaseModel):
    def __init__(self, model_params, fit_params: Optional[Dict]):
        self.model_params = model_params
        self.fit_params = fit_params
        if self.fit_params is None:
            self.fit_params = {}

    def build_model(self):
        self.model = HistGradientBoostingClassifier(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            **self.fit_params
        )
        return self.model

    def predict(self, est, valid_x):
        preds = est.predict_proba(valid_x)[:, 1]
        return preds