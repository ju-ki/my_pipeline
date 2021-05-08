import numpy as np


class BaseModel(object):
    def build_model(self, **kwargs):
        raise NotImplementedError

    def fit(self, train_x, train_y, valid_x, valid_y):
        raise NotImplementedError

    def predict(self, model, valid_x):
        raise NotImplementedError

    def run(self, name, train_x, train_y, cv, metrics, n_splits, seeds):
        oof_seeds = []
        score_seeds = []
        models = {}
        for seed in seeds:
            oof = []
            va_idxes = []
            scores = []
            fold_idx = cv(train_x.values, train_y.values,
                          n_splits=n_splits, random_state=seed)

            for cv_num, (tr_idx, va_idx) in enumerate(fold_idx):
                tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
                tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]
                va_idxes.append(va_idx)

                model = self.build_model()
                model = self.fit(tr_x, tr_y, va_x, va_y)
                model_name = f"{name}_SEED{seed}_FOLD{cv_num}_model"
                models[model_name] = model

                pred = self.predict(self.model, va_x)
                oof.append(pred)

                score = metrics(va_y, pred)
                scores.append(score)
                print(
                    f"SEED:{seed}, FOLD:{cv_num} ----------------> val_score:{score:.4f}")
            va_idxes = np.concatenate(va_idxes)
            oof = np.concatenate(oof)
            order = np.argsort(va_idxes)
            oof = oof[order]
            oof_seeds.append(oof)
            score_seeds.append(np.mean(scores))

        oof = np.mean(oof_seeds, axis=0)
        print(f"FINISHED| model:{name} score:{metrics(train_y, oof):.4f}\n")
        return oof, models

    def inference(self, test_x, models):
        preds = []
        for name, est in models.items():
            print(f"{name} ------->")
            _pred = est.predict(test_x)
            preds.append(_pred)
        preds = np.mean(preds, axis=0)
        return preds
