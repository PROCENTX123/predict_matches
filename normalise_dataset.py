import pandas as pd


def normalize_column(column_values):
    # minimax
    x = column_values.to_numpy()
    return (x-min(x))/(max(x)-min(x))


def normalize_dataset_values(dataset, y_column_name="Y"):
    normalized_dict = {}
    for column_name in dataset.columns.tolist():
        if column_name != y_column_name:
            normalized_dict[column_name] = normalize_column(dataset[column_name])
        else:
            normalized_dict[column_name] = dataset[column_name]
    return pd.DataFrame(normalized_dict)

