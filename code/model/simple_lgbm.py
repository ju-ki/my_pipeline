import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgbm
from lightgbm import LGBMModel
from sklearn.metrics import mean_squared_error
from util import timer
from lgbm import BaseModel


class MyAllLGBMModel:
    def __init__(self, params=None, name=None, cv=None, X_train=None, y_train=None, X_test=None, metrics=None, seeds=None):
        # LGBMのモデル
        # ref: https://signate.jp/competitions/402/discussions/lgbm-baseline-except-text-vs-include-text-lb07994-1
        """
        クラスで学習、推論、特徴量重要度の可視化
        cvは次のようにlistで返す

        group = train_df["art_series_id"]
        def make_gkf(X, y, n_splits=5, random_state=0):
          gkf = GroupKFold(n_splits=n_splits, random_state=random_state)
          return list(gkf.split(X, y, group))

        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.params = params
        self.metrics = metrics
        self.name = name
        self.cv = cv
        self.oof = None
        self.preds = None
        self.seeds = seeds
        self.models = {}

        self.train_x = self.X_train.values
        self.train_y = self.y_train.values
        self.fold_idx = self.cv(
            self.train_x, self.train_y, n_splits=5, random_state=0)

    def build_model(self):
        model = LGBMModel(**self.params)
        return model

    def predict_cv(self):
        oof_seeds = []
        scores_seeds = []
        for seed in self.seeds:
            oof = []
            va_idxes = []
            scores = []

            for cv_num, (tr_idx, va_idx) in enumerate(self.fold_idx):
                tr_x, va_x = self.train_x[tr_idx], self.train_x[va_idx]
                tr_y, va_y = self.train_y[tr_idx], self.train_y[va_idx]
                va_idxes.append(va_idx)
                model = self.build_model()

                model.fit(tr_x, tr_y,
                          eval_set=[[va_x, va_y]],
                          early_stopping_rounds=100,
                          verbose=False
                          )
                model_name = f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl"
                self.models[model_name] = model

                pred = model.predict(va_x)
                oof.append(pred)

                score = self.get_score(va_y, pred)
                scores.append(score)
                print(
                    f"SEED:{seed}, FOLD:{cv_num} ------------> val_score:{score}")

            va_idxes = np.concatenate(va_idxes)
            oof = np.concatenate(oof)
            order = np.argsort(va_idxes)
            oof = oof[order]
            oof_seeds.append(oof)
            scores_seeds.append(np.mean(scores))

        oof = np.mean(oof_seeds, axis=0)
        self.oof = oof
        print(
            f"FINISHED| model:{self.name} score:{self.get_score(self.y_train, oof)}\n")
        return oof

    def inference(self):
        preds_seeds = []
        for seed in self.seeds:
            preds = []
            X_test = self.X_test.values

            for cv_num in range(5):
                print(f"INFERENCE| -SEED:{seed}, FOLD:{cv_num}")
                model_name = f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl"
                model = self.models[model_name]

                pred = model.predict(X_test)
                preds.append(pred)

            preds = np.mean(preds, axis=0)
            preds_seeds.append(preds)

        preds = np.mean(preds_seeds, axis=0)
        self.preds = preds
        return preds

    def visualize_feature_importance(self):
        feature_importance_df = pd.DataFrame()
        for i, (tr_idx, va_idx) in enumerate(self.fold_idx):
            tr_x, va_x = self.X_train.values[tr_idx], self.X_train.values[va_idx]
            tr_y, va_y = self.y_train.values[tr_idx], self.y_train.values[va_idx]
            model = self.build_model()
            model.fit(tr_x, tr_y,
                      eval_set=[[va_x, va_y]],
                      early_stopping_rounds=100,
                      verbose=False)
            _df = pd.DataFrame()
            _df["feature_importance"] = model.feature_importances_
            _df["column"] = self.X_train.columns
            _df["fold"] = i + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, _df], axis=0, ignore_index=True)
        order = feature_importance_df.groupby("column").sum()[["feature_importance"]].sort_values(
            "feature_importance", ascending=False).index[:50]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df, y="column",
                      x="feature_importance", order=order, ax=ax, palette="viridis")
        fig.tight_layout()
        ax.grid()
        ax.set_title("feature importance top50")
        fig.tight_layout()
        plt.show()
        return fig, feature_importance_df

    def get_score(self, y_true, y_pred):
        score = self.metrics(y_true, y_pred)
        return score

    class MyLGBMModel(BaseModel):

        # ref:https://qiita.com/tubo/items/f83a97f2488cc1f40088 tuboさんのベースラインから
        #   :https://signate.jp/competitions/402/discussions/lgbm-baseline-except-text-vs-include-text-lb07994-1　masatoさんのベースラインから

    def __init__(self, model_params, fit_params):
        self.model_params = model_params
        self.fit_params = fit_params
        self.models = []
        self.model = None

    def build_model(self):
        self.model = LGBMModel(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x, valid_y):
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[[valid_x, valid_y]],
            **self.fit_params
        )
        self.models.append(self.model)
        return self.model

    def predict(self, model, valid_x):
        return self.model.predict(valid_x)

    def visualize_feature_importance(self, train_x, train_y, cv, num=50):
        self.vis_model = self.build_model()
        feature_importance_df = pd.DataFrame()
        for cv_num, (tr_idx, va_idx) in enumerate(cv(train_x.values, train_y.values, n_splits=5)):
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]
            self.vis_model.fit(tr_x, tr_y,
                               eval_set=[[va_x, va_y]],
                               **self.fit_params)
            _df = pd.DataFrame()
            _df["feature_importance"] = self.vis_model.feature_importances_
            _df["columns"] = train_x.columns
            _df["fold"] = cv_num + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, _df], axis=0, ignore_index=True)
        order = feature_importance_df.groupby("columns").sum()[["feature_importance"]].sort_values(
            "feature_importance", ascending=False).index[:num]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df, y="columns",
                      x="feature_importance", order=order, ax=ax, palette="viridis")
        fig.tight_layout()
        ax.grid()
        ax.set_title(f"feature_importance_TOP{num}")
        fig.tight_layout()
        plt.show()
        return fig, feature_importance_df
