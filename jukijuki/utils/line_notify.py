import requests


class LineController:
    def __init__(self, path, config):
        self.path = path
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
        with open(self.path, "r") as f:
            line_token = f.readline()
        endpoint = 'https://notify-api.line.me/api/notify'
        message = f"[{env}]{message}"
        payload = {'message': message}
        headers = {'Authorization': 'Bearer {}'.format(line_token)}
        requests.post(endpoint, data=payload, headers=headers)
