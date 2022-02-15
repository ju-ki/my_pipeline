import os
import sys
import numpy as np
import pandas as pd
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

    def run(self, name: str, train_x: pd.DataFrame, train_y: np.ndarray, cv, metrics, logger=None, output_dir: str = "./") -> np.ndarray:
        models = {}
        oof = []
        va_idxes = []
        scores = []

        for cv_num, (tr_idx, va_idx) in tqdm(enumerate(cv)):
            if logger is None:
                print(decorate("fold {}".format(cv_num + 1) + " is starting"))
            else:
                logger.info(decorate("fold {}".format(cv_num + 1) + " is starting"))
            tr_x, va_x = train_x.loc[tr_idx], train_x.loc[va_idx]
            tr_y, va_y = train_y.loc[tr_idx], train_y.loc[va_idx]
            va_idxes.append(va_idx)

            model = self.build_model()
            model = self.fit(tr_x, tr_y, va_x, va_y)
            model_name = f"{name}_FOLD{cv_num}_model"
            self.save(output_dir, model_name)
            models[model_name] = model

            pred = self.predict(self.model, va_x)
            oof.append(pred)

            score = metrics(va_y, pred)
            scores.append(score)
            if logger is None:
                print(f"FOLD:{cv_num} ----------------> val_score:{score:.4f}")
            else:
                logger.info(f"FOLD:{cv_num} ----------------> val_score:{score:.4f}")

        va_idxes = np.concatenate(va_idxes)
        oof = np.concatenate(oof)
        order = np.argsort(va_idxes)
        oof = oof[order]
        if logger is None:
            print(f"FINISHED| model:{name} score:{metrics(train_y, oof):.4f}\n")
        else:
            logger.info(f"FINISHED| model:{name} score:{metrics(train_y, oof):.4f}\n")
        return oof, models, va_idxes

    def inference(self, test_x: pd.DataFrame, models) -> np.ndarray:
        preds = []
        for name, est in models.items():
            print(f"{name} ------->")
            _pred = est.predict(test_x)
            preds.append(_pred)
        preds = np.mean(preds, axis=0)
        return preds

    def save(self, filepath: str, model_name: str):
        return Util.dump(self.model, filepath + model_name + ".pkl")

    def load(self, filepath: str, model_name: str):
        self.model = Util.load(filepath + model_name + ".pkl")

    def make_oof(self, df: pd.DataFrame, filepath: str, is_pickle=False):
        return Util.dump_df(df, filepath + "oof", is_pickle=is_pickle)
