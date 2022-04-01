import os
import sys
import pandas as pd
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.timer import Timer
from utils.util import Util, decorate


class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def fit_transform(self, input_df: pd.DataFrame, y=None):
        self.fit(input_df, y=y)
        return self.transform(input_df)


class WrapperBlock(AbstractBaseBlock):
    def __init__(self, function):
        self.function = function

    def transform(self, input_df):
        return self.function(input_df)


def run_blocks(input_df: pd.DataFrame, blocks: List, y=None, preprocess_block=None,
               logger=None, filepath: str = "./", task: str = "train", save_feature: bool = False) -> pd.DataFrame:
    """
    Args:
        input_df (pd.DataFrame): original DataFrame
        blocks (List): function block
        y (_type_, optional): _description_. Defaults to None.
        preprocess_block (_type_, optional): if need preprocessing for example fillna, you need set function of preporcessing
        logger (_type_, optional): if is not None, output log fie
        filepath (str, optional): output feature block as pkl. Defaults to "./".
        task (str, optional): _description_. Defaults to "train".
        save_feature; create feature as pkl. default=False

    Returns:
        pd.DataFrame: feature engined feature
    """
    out_df = pd.DataFrame()
    if preprocess_block is not None:
        input_df = preprocess_block(input_df)
    _input_df = input_df.copy()

    if save_feature and not os.path.isdir(filepath + "features/"):
        os.makedirs(filepath + "features")

    print(decorate(f"start create block for {task}"))

    with Timer(logger=logger, prefix=f'create {task} block'):
        for block in blocks:
            if save_feature:
                try:
                    file_name = os.path.join(filepath + "features/", f"{task}_{block.__class__.__name__}_{str(block.cols)}.pkl")
                except:
                    file_name = os.path.join(filepath + "features/", f"{task}_{block.__class__.__name__}.pkl")

            with Timer(logger=logger, prefix='\t- {}'.format(str(block))):
                if save_feature and os.path.isfile(file_name):
                    out_i = Util.load(file_name)
                else:
                    if task == "train":
                        out_i = block.fit(_input_df)
                        if save_feature:
                            Util.dump(out_i, file_name)
                    else:
                        out_i = block.transform(_input_df)
                        if save_feature:
                            Util.dump(out_i, file_name)

            assert len(input_df) == len(out_i), block
            name = block.__class__.__name__
            out_df = pd.concat([out_df, out_i.add_suffix(f'@{name}')], axis=1)

    return out_df