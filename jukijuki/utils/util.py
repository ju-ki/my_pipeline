import os
import sys
import pandas as pd
import subprocess
import joblib


class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def save_csv(cls, df, path, name):
        df.to_csv(path + name + ".csv", index=False)

    @classmethod
    def dump_df(cls, df, path, is_pickle=False):
        if is_pickle:
            df.to_pickle(path)
        else:
            df.to_csv(path, index=False)

    @classmethod
    def load_df(cls, path, is_pickle=False):
        if is_pickle:
            return pd.read_pickle(path)
        else:
            return pd.read_csv(path)


def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = '*' * 20

    return ' '.join([decoration, str(s), decoration])


def set_environment() -> bool:
    """[summary]
    今いる環境をboolで返す関数
    Returns:
        bool: [current environment(colab, kaggle, local)]

    Example usage:
       IN_KAGGLE, IN_COLAB, LOCAL = set_environment()
    """
    IN_COLAB = 'google.colab' in sys.modules
    IN_KAGGLE = 'kaggle_web_client' in sys.modules
    LOCAL = not (IN_KAGGLE or IN_COLAB)
    print(f'IN_COLAB:{IN_COLAB}, IN_KAGGLE:{IN_KAGGLE}, LOCAL:{LOCAL}')
    return IN_COLAB, IN_KAGGLE, LOCAL


def make_exp_output_directory(Config):
    assert hasattr(Config, "model_dir"), "Please model_dir attribute"
    assert hasattr(Config, "exp_name"), "Please create exp_name attribute"
    import requests

    def get_exp_name():
        return requests.get("http://172.28.0.2:9000/api/sessions").json()[0]["name"].split("_")[0]
    if not os.path.isdir(Config.model_dir + get_exp_name()):
        try:
            subprocess.run(f"mkdir {Config.model_dir + get_exp_name()}", shell=True, stdout=subprocess.PIPE, check=True)
            Config.model_dir = Config.model_dir + get_exp_name()
            Config.exp_name = get_exp_name()
            print(f"Created {get_exp_name()} folder")
        except subprocess.CalledProcessError:
            print("Please check your folder")
    else:
        Config.model_dir = Config.model_dir + get_exp_name() + "/"
        Config.exp_name = get_exp_name()
        print(f"Already such {Config.model_dir} folder created!")