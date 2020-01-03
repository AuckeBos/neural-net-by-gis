from datetime import datetime


def get_progress(percentage):
    """
    Return progressbar
    :param percentage: Percentage done
    :return: Progressbar (string)
    """
    max_width = 20
    progress = int(percentage * max_width)
    leftover = max_width - progress
    return f'[{"=" * progress}>{" " * leftover}]'


def log(text, end='\n'):
    """
    Create log in standard format and print it
    :param text: Text to log
    :param end: Character to finish log with
    """
    print(f'[INFO] - {text}', end=end)


def get_time():
    """
    Get current time in specific format, used for filenames
    :return: Current time as string in specific format
    """
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
