import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from .base import BaseModel
from utils.util import decorate


class MyTabNetModel(BaseModel):
    def __init__(self, model_params, fit_params):
        self.model_params = model_params
        self.fit_params = fit_params
        self.model = None

    def build_model(self):
        self.model = TabNetRegressor(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_name=["train", "valid"],
            **self.fit_params
        )
        return self.model

    def predict(self, model, valid_x):
        preds = self.model.predict(valid_x)
        return preds


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