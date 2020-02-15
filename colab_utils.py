import pandas as pd
from google.colab import files


def read_file_colab(file_name):
    print("I need {} file".format(file_name))
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError as e:
        uploaded = files.upload()
    return pd.read_csv(file_name)
