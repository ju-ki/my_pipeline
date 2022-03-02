import torch


class EarlyStopping:

    def __init__(self, patience=2, logger=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.logger = logger

    def __call__(self, loss, model, preds, path, minimize=True):
        if minimize:
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(model, preds, path)
            elif loss < self.best_score:
                self.logger.info(
                    f'Loss decreased {self.best_score:.5f} -> {loss:.5f}')
                self.best_score = loss
                self.counter = 0
                self.save_checkpoint(model, preds, path)
            else:
                self.counter += 1
                if self.counter > self.patience:
                    self.stop = True
        else:
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(model, preds, path)
            elif loss > self.best_score:
                self.logger.info(
                    f'Loss increased {self.best_score:.5f} --> {loss:.5f}')
                self.best_score = loss
                self.counter = 0
                self.save_checkpoint(model, preds, path)
            else:
                self.counter += 1
                if self.counter > self.patience:
                    self.stop = True

    def save_checkpoint(self, model,  preds, path):
        save_list = {'model': model.state_dict(),
                     'preds': preds}
        SAVE_PATH = path
        torch.save(save_list, SAVE_PATH)
