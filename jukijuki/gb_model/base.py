import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from IPython.display import display
from tqdm.auto import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import decorate, Util


class BaseModel(object):

    def build_model(self, **kwargs):
        raise NotImplementedError

    def fit(self, train_x, train_y, valid_x, valid_y):
        raise NotImplementedError

    def predict(self, model, valid_x):
        raise NotImplementedError

    def run(self, name: str, train_x: pd.DataFrame, train_y: Union[pd.Series, np.ndarray], cv: List[Tuple], metrics, logger = None, output_dir: str = "./", save_model: bool=False):
        """
        Parameters
        ---------------------------
        name: str
          experiment name
        train_x: pd.DataFrame
           train set
        train_y: pd.Series or np.np.ndarray
          train target
        cv: List[Tuple]
          set cross validation
        metrics:
          set metrics for evaluating score
         logger:
          output log files
          None -> print log
         output_dir: str
          output model as pkl
          save_model: bool
          whether create pretrained model or not. default=False

        Example
        -------------------------------

          def make_kfold(X, y):
              kf = KFold(n_splits=5, shuffle=True, random_state=42)
              return list(kf.split(X, y))

         def score_rmse(y_true, y_pred):
             touscore = mean_square_error(y_true, y_pred, is_squared=False)
             return score

        cv = make_kfold(X, y)

        ### set params for model
        my_model = MyModel(model_params=lgbm_params, fit_params=fit_params)

        ### learning
        my_model.run(name='baseline', train_x=X, train_y=y, cv=cv, metrics=score_rmse, logger=None, output_dir="./)

        ### make oof
        my_model.make_oof()

        ### inference
        my_model.inference(test_x=test_x)

        ### make submission
        my_model.make_submission()

        ### plot oof and pred
        my_model.plot_oof_pred_target

        ### display oof and pred
        my_model.debug_oof_pred(num=5)

        """

        self.name = name
        self.output_dir = output_dir
        self.train_y = train_y
        self.models = {}
        self.oof = []
        va_idxes = []
        scores = []

        for cv_num, (tr_idx, va_idx) in tqdm(enumerate(cv)):
            if logger is None:
                print(decorate("fold {}".format(cv_num + 1) + " is starting"))
            else:
                logger.info(decorate("fold {}".format(cv_num + 1) + " is starting"))
            tr_x, va_x = train_x.loc[tr_idx], train_x.loc[va_idx]
            tr_y, va_y = self.train_y[tr_idx], self.train_y[va_idx]
            va_idxes.append(va_idx)

            model = self.build_model()
            model = self.fit(tr_x, tr_y, va_x, va_y)
            model_name = f"{name}_FOLD{cv_num}_model"
            if save_model:
                self.save(self.output_dir, model_name)
            self.models[model_name] = model

            pred = self.predict(model, va_x)
            self.oof.append(pred)

            score = metrics(va_y, pred)
            scores.append(score)
            if logger is None:
                print(f"FOLD:{cv_num + 1} ----------------> val_score:{score:.4f}")
            else:
                logger.info(f"FOLD:{cv_num + 1} ----------------> val_score:{score:.4f}")

        va_idxes = np.concatenate(va_idxes)
        self.oof = np.concatenate(self.oof)
        order = np.argsort(va_idxes)
        self.oof = self.oof[order]
        if logger is None:
            print(f"FINISHED| model:{name} score:{metrics(self.train_y, self.oof):.4f}\n")
        else:
            logger.info(f"FINISHED| model:{name} score:{metrics(self.train_y, self.oof):.4f}\n")

    def inference(self, test_x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self.preds = []
        for name, est in self.models.items():
            print(f"{name} ------->")
            _pred = self.predict(est, test_x)
            self.preds.append(_pred)
        self.preds = np.mean(self.preds, axis=0)

    def save(self, filepath: str, model_name: str):
        return Util.dump(self.model, filepath + model_name + ".pkl")

    def load(self, filepath: str, model_name: str):
        self.model = Util.load(filepath + model_name + ".pkl")

    def make_oof(self):
        self.oof = pd.DataFrame({
            "oof": self.oof
        })
        return Util.save_csv(self.oof, self.output_dir, self.name + "oof")

    def make_submission(self):
        self.sub_df = pd.DataFrame()
        self.sub_df["target"] = self.preds
        return Util.save_csv(self.sub_df, self.output_dir,  self.name + "sub")

    def plot_oof_pred_target(self):
        sns.set()
        plt.figure(figsize=(20, 7))
        sns.distplot(self.train_y, label="True Target")
        sns.distplot(self.oof.values, label=self.name + "_oof")
        sns.distplot(self.preds, label=self.name + "_pred")
        plt.legend()
        plt.show()

    def debug_oof_pred(self, num=5):
        display(self.oof.head(num))
        display(self.sub_df.head(num))
