import os
import sys
import torch
import random
import pandas as pd
import numpy as np
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


def seed_everything(SEED):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def set_environment(Config) -> bool:
    """[summary]
    今いる環境をboolで返す関数
    Returns:
        bool: [current environment(colab, kaggle, local)]
    """
    IN_COLAB = 'google.colab' in sys.modules
    IN_KAGGLE = 'kaggle_web_client' in sys.modules
    LOCAL = not (IN_KAGGLE or IN_COLAB)
    Config.IN_COLAB = IN_COLAB
    Config.IN_KAGGLE = IN_KAGGLE
    Config.LOCAL = LOCAL
    print(f'IN_COLAB:{IN_COLAB}, IN_KAGGLE:{IN_KAGGLE}, LOCAL:{LOCAL}')


def create_folder(Config):
    set_environment(Config)
    assert hasattr(Config, "competition_name"), "Please create competition_name attribute"
    assert hasattr(Config, "exp_name"), "Please create exp_name attribute"
    assert hasattr(Config, "IN_COLAB"), "Please execute set_environment"
    assert hasattr(Config, "IN_KAGGLE"), "Please excute set_environment"

    if Config.IN_COLAB:
        import requests
        os.chdir("/content/drive/MyDrive/")
        if not os.path.isdir(Config.competition_name):
            os.makedirs(Config.competition_name)

        def get_exp_name():
            return requests.get("http://172.28.0.2:9000/api/sessions").json()[0]["name"].split("_")[0]

        Config.exp_name = get_exp_name()
        print(decorate(Config.exp_name))
        os.chdir(f"/content/drive/MyDrive/{Config.competition_name}/")
        INPUT = f"/content/drive/MyDrive/{Config.competition_name}/"
        DATA = os.path.join(INPUT, "input/")
        EXTERNAL = os.path.join(DATA, "external/")
        LOG = os.path.join(INPUT, "log/")
        OUTPUT = os.path.join(INPUT, "output/")
        EXP = os.path.join(OUTPUT, Config.exp_name + "/")
        Config.input_dir = DATA
        Config.external_dir = EXTERNAL
        Config.output_dir = OUTPUT
        Config.model_dir = EXP
        Config.log_dir = LOG
        for d, name in zip([LOG, OUTPUT, EXP, DATA, EXTERNAL], ["log_dir", "output_dir", "model_dir", "input_dir", "external_dir"]):
            if not os.path.isdir(d):
                print(f"{name}: making {d}")
                os.makedirs(d, exist_ok=True)
            else:
                print(f"{name}: already created{d}")
        file_name = os.path.join(Config.log_dir, Config.exp_name + ".log")
        if os.path.isfile(file_name):
            os.remove(file_name)
            print(f"recreate {Config.exp_name}.log file")
    elif Config.IN_KAGGLE:
        Config.input_dir = f"../input/{Config.competition_name}/"
        Config.output_dir = "./"
        Config.model_dir = "./"
        Config.log_dir = "./"