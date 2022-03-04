import os
import json
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
