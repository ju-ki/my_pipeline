import os
import sys
import json
import glob
import subprocess


def create_new_dataset(dataset_name, upload_dir):
    """[summary]

    Args:
        dataset_name ([string]): set dataset name for kaggle display
        upload_dir ([string(path)]): set path where the target folder is located 
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    dataset_metadata = {}
    dataset_metadata['id'] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
    dataset_metadata['licenses'] = [{'name': 'CC0-1.0'}]
    dataset_metadata['title'] = dataset_name
    with open(os.path.join(upload_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(dataset_metadata, f, indent=4)
    api = KaggleApi()
    api.authenticate()
    api.dataset_create_new(
        folder=upload_dir, convert_to_csv=False, dir_mode='tar')


def set_kaggle_info():
    """[summary]
    set your kaggle information to Google Colaboratory
    """
    f = open("/content/drive/MyDrive/jukiya/kaggle.json", 'r')
    json_data = json.load(f)
    os.environ["KAGGLE_USERNAME"] = json_data["username"]
    os.environ["KAGGLE_KEY"] = json_data["key"]


def set_kaggle_api(debug_command=False):
    """
     Google Colaboratoryでkaggle apiを使用するための関数
    """
    os.chdir("/content/drive/MyDrive/jukiya/")
    try:
        o = subprocess.run("pip install kaggle", shell=True, stdout=subprocess.PIPE, check=True)
        print(o.stdout.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print("Not kaggle.json file",  e.stderr)
    try:
        o = subprocess.run("mkdir -p ~/kaggle", shell=True,  stdout=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("Error",  e.stderr)
    try:
        o = subprocess.run("cp kaggle.json ~/.kaggle/", shell=True, stdout=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("Error cp command:",  e.stderr)
    try:
        o = subprocess.run("chmod 600 /root/.kaggle/kaggle.json", shell=True, stdout=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("Error chmod command:",  e.stderr)
    if debug_command:
        try:
            result = subprocess.run("kaggle competitions list", shell=True, stdout=subprocess.PIPE, check=True)
            print(result.stdout.decode("utf-8"))
        except subprocess.CalledProcessError as e:
            print("Error kaggle command:", e.stderr)
    print("Completed set kaggle api!")
    os.chdir("/content/")


def set_environment(competition_name: str) -> bool:
    """[summary]
    主にGoogle Colaboratoryで必要なライブラリやコンペデータのダウンロードをする関数
    Args:
        competition_name (str): [Name of the competition you want to download]

    Returns:
        bool: [current environment(colab, kaggle, local)]
    """
    IN_COLAB = 'google.colab' in sys.modules
    IN_KAGGLE = 'kaggle_web_client' in sys.modules
    LOCAL = not (IN_KAGGLE or IN_COLAB)
    print(f'IN_COLAB:{IN_COLAB}, IN_KAGGLE:{IN_KAGGLE}, LOCAL:{LOCAL}')
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        set_kaggle_api()
        input_dir = f"/content/drive/MyDrive/{competition_name}/data/input/"
        data_path = glob.glob(os.path.join(input_dir, "*submission.csv"))
        if len(data_path) == 0 or not os.path.isfile(data_path[0]):
            os.chdir(input_dir)
            try:
                result = subprocess.run(f"kaggle competitions download -c {competition_name}", shell=True, stdout=subprocess.PIPE, check=True)
                print(result.stdout.decode("utf-8"))
            except subprocess.CalledProcessError as e:
                print("Error competition name:", e.stderr)
            try:
                o = subprocess.run("unzip '*.zip'", shell=True, stdout=subprocess.PIPE, check=True) 
                print(o.stdout.decode("utf-8"))
            except subprocess.CalledProcessError as e:
                print("Error unzip command:", e.stderr)
            try:
                o = subprocess.run("rm *.zip", shell=True, stdout=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                print("Error rm command:", e.stderr)
        os.chdir(f"/content/drive/MyDrive/{competition_name}/")
        try:
            o = subprocess.run("pip install --quiet -r requirements.txt", shell=True, stdout=subprocess.PIPE, check=True)
            print(o.stdout.decode("utf-8"))
            print("Completed installing Library!")
        except subprocess.CalledProcessError as e:
            print("Error install library:", e.stderr)
        os.chdir("/content/")
    return IN_COLAB, IN_KAGGLE, LOCAL
