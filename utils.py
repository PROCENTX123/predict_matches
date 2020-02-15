import pandas as pd


def read_file_local(file_name):
    print("I need {} file".format(file_name))
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError as e:
        print("File not found")
        raise

