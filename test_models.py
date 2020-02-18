from train_models import split_dataset, linear_regression, linear_regression_rfe, split_scikit, logistic_regression_rfe, \
    decision_tree, random_forest
from training_metrics import class_report
from utils import read_file_local


def test_split_dataset():
    dataset = read_file_local('data_files/original_ds_not_normalized.csv')
    print(split_dataset(dataset))


def test_lin_regr():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.3
    )
    train_predict, test_predict = linear_regression(train_x, test_x, train_y, test_y)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))


def test_lin_regr_rfe():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.3
    )
    train_predict, test_predict = linear_regression_rfe(train_x, test_x, train_y, test_y, 2)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))


def test_log_regr_rfe():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.4
    )
    train_predict, test_predict = logistic_regression_rfe(train_x, test_x, train_y, test_y, 2)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))


def test_decision_tree():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.4
    )
    train_predict, test_predict = decision_tree(train_x, test_x, train_y, test_y, 26)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))

def test_random_forest():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.4
    )
    random_forest(train_x, test_x, train_y, test_y, 5)

test_lin_regr_rfe()
test_log_regr_rfe()
test_decision_tree()
# test_random_forest()
test_lin_regr()