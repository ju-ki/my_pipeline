import requests


def send_line_notification(message: str, token: str, config = None):
    """[summary]

    Args:
        message ([string]): [content that notify for your line]
        token ([string]): [your line token]
        env ([string]): [set your environment (example:Kaggle Colab Local)]
    """
    if config.IN_COLAB:
        env = "colab"
    elif config.IN_KAGGLE:
        env = "kaggle"
    elif config.IN_LOCAL:
        env = "local"
    line_token = token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = f"[{env}]{message}"
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)
