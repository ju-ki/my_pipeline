import datetime
import logging
import sys
import os
import numpy as np
import pandas as pd
import inspect
import random
import torch
import tensorflow as tf
import yaml
import joblib
from time import time
from contextlib import contextmanager


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHEDSEED"] = str(seed)
    np.random.seed(seed)
    tf.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# coding: UTF-8
CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
SUB_DIR_NAME = yml['SETTING']['SUB_DIR_NAME']

# tensorflowとloggingのcollisionに対応
try:
    import absl.logging
    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def dump_df_pickle(cls, df, path):
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)


class Logger:
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(
            os.path.join(path, 'Experiment.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        self.general_logger.info(
            '[{}] - {}'.format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class Submission:

    @classmethod
    def create_submission(cls, run_name, path, sub_y_column):
        logger = Logger(path)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(RAW_DATA_DIR_NAME + 'sample_submission.csv')
        pred = Util.load_df_pickle(path + f'{run_name}-pred.pkl')
        submission[sub_y_column] = pred
        submission.to_csv(path + f'{run_name}_submission.csv', index=False)

        logger.info(f'{run_name} - end create submission')


class AbstractBaseBlock:
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()


class WrapperBlock(AbstractBaseBlock):
    def __init__(self, function):
        self.function = function

    def transform(self, input_df):
        return self.function(input_df)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


def param_to_name(params: dict, key_sep='_', key_value_sep='=') -> str:
    """
    dict を `key=value` で連結した string に変換します.
    Args:
        params:
        key_sep:
            key 同士を連結する際に使う文字列.
        key_value_sep:
            それぞれの key / value を連結するのに使う文字列.
            `"="` が指定されると例えば { 'foo': 10 } は `"foo=10"` に変換されます.
    Returns:
        文字列化した dict
    """
    sorted_params = sorted(params.items())
    return key_sep.join(map(lambda x: key_value_sep.join(map(str, x)), sorted_params))


def cachable(function):
    attr_name = '__cachefile__'

    def wrapper(*args, **kwrgs):
        force = kwrgs.pop('force', False)
        call_args = inspect.getcallargs(function, *args, **kwrgs)

        arg_name = param_to_name(call_args)
        name = attr_name + arg_name

        use_cache = hasattr(function, name) and not force

        if use_cache:
            cache_object = getattr(function, name)
        else:
            print('run')
            cache_object = function(*args, **kwrgs)
            setattr(function, name, cache_object)

        return cache_object

    return wrapper


@cachable
def read_csv(name, INPUT_PATH):

    if '.csv' not in name:
        name = name + '.csv'

    return pd.read_csv(os.path.join(INPUT_PATH, name))


@contextmanager
def timer(logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None):

    if prefix:
        format_str = str(prefix) + format_str
    if suffix:
        format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)
