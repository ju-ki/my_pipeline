import requests


def get_token(config, use_line=True, use_wandb=False):
    line_token = ""
    wandb_token = ""
    if config.IN_KAGGLE:
        path = "../input/my_token/"
    elif config.IN_COLAB:
        path = "/content/drive/MyDrive/jukiya/"
    if use_line:
        with open(path+"line_token.txt", "r") as f:
            line_token = f.readline()
    if use_wandb:
        with open(path+"line_token.txt", "r") as f:
            wandb_token = f.readline()
    return line_token, wandb_token


class LineController:
    def __init__(self, token, config):
        self.token = token
        self.config = config

    def send_line_notification(self, message: str):
        """[summary]
        Args:
           message ([string]): [content that notify for your line]
           token ([string]): [your line token]
           env ([string]): [set your environment (example:Kaggle Colab Local)]
        """
        if self.config.IN_COLAB:
            env = "colab"
        elif self.config.IN_KAGGLE:
            env = "kaggle"
        elif self.config.IN_LOCAL:
            env = "local"
        endpoint = 'https://notify-api.line.me/api/notify'
        message = f"[{env}]{message}"
        payload = {'message': message}
        headers = {'Authorization': 'Bearer {}'.format(self.token)}
        requests.post(endpoint, data=payload, headers=headers)
