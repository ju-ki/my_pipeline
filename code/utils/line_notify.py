import requests


def send_line_notification(message, token, env):
    """[summary]

    Args:
        message ([string]): [content that notify for your line]
        token ([string]): [your line token]
        env ([string]): [set your environment (example:Kaggle Colab Local)]
    """
    line_token = token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = f"[{env}]{message}"
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)
