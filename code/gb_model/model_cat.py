from catboost import Pool, CatBoost
from base import BaseModel


class MyCatModel(BaseModel):
    def __init__(self, model_params, fit_params):
        self.model_params = model_params
        self.fit_params = fit_params

    def build_model(self):
        model = CatBoost(self.model_params)
        return model

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_pool = Pool(train_x, train_y)
        valid_pool = Pool(valid_x, valid_y)
        self.model = self.build_model()
        self.model.fit(train_pool,
                       plot=False,
                       use_best_model=True,
                       eval_set=[valid_pool],
                       **self.fit_params
                       )
        return self.model

    def predict(self, model, valid_x):
        return model.predict(valid_x)
