from datetime import datetime


def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_timestamped_string(pattern: str):
    return pattern.format(get_current_timestamp())
