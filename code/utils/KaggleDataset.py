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


def set_kaggle_api():
    """
     Google Colaboratoryでkaggle apiを使用するための関数
    """
    os.chdir("/content/drive/MyDrive/jukiya/")
    try:
        subprocess.check_output("pip install kaggle", shell=True)
        subprocess.check_output("mkdir -p ~/kaggle", shell=True)
        subprocess.check_output("cp kaggle.json ~/.kaggle/", shell=True)
        subprocess.check_output(
            "chmod 600 /root/.kaggle/kaggle.json", shell=True)
        result = subprocess.run("kaggle competitions list",
                                shell=True, stdout=subprocess.PIPE)
        print(result.stdout.decode("utf-8"))
    except subprocess.CalledProcessError:
        print("Error: Pleasse check command or library name")
    os.chdir("/content/")
