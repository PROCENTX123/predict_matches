from sklearn import metrics

from train_models import split_dataset, linear_regression, linear_regression_rfe, split_scikit, logistic_regression_rfe, \
    decision_tree, random_forest, xgboost_forest, catboost_forest, lgbm_forest
from training_metrics import class_report, plot_roc_auc, conf_matrix
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
    print(conf_matrix(test_predict, test_y))

def test_lin_regr_rfe():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.3
    )
    train_predict, test_predict = linear_regression_rfe(train_x, test_x, train_y, test_y, 2)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))
    print(conf_matrix(test_predict, test_y))
def test_log_regr_rfe():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.4
    )
    train_predict, test_predict, train_predict_proba, test_predict_proba = logistic_regression_rfe(train_x, test_x, train_y, test_y, 2)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))
    plot_roc_auc(test_predict_proba, test_y)
    plot_roc_auc(train_predict_proba, train_y)
    print(conf_matrix(test_predict,test_y))

def test_decision_tree():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.4
    )
    train_predict, test_predict, train_predict_proba, test_predict_proba = decision_tree(train_x, test_x, train_y, test_y, 26)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))
    plot_roc_auc(test_predict_proba, test_y)
    plot_roc_auc(train_predict_proba, train_y)
    print(conf_matrix(test_predict, test_y))



def test_random_forest():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.6
    )
    train_predict, test_predict, train_predict_proba, test_predict_proba = random_forest(train_x, test_x, train_y, test_y)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))
    plot_roc_auc(test_predict_proba, test_y)
    plot_roc_auc(train_predict_proba, train_y)
    print(conf_matrix(test_predict, test_y))


def test_xgb_forest():
    dataset = read_file_local('data_files/original_ds_not_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.6
    )
    train_predict, test_predict, train_predict_proba, test_predict_proba = xgboost_forest(train_x, test_x, train_y, test_y)
    # print(class_report(train_predict, train_y))
    # print(class_report(test_predict, test_y))
    # plot_roc_auc(test_predict_proba, test_y)
    # plot_roc_auc(train_predict_proba, train_y)
    print(conf_matrix(test_predict, test_y))



def test_catboost_forest():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.6
    )
    train_predict, test_predict, train_predict_proba, test_predict_proba = catboost_forest(train_x, test_x, train_y, test_y)
    # print(class_report(train_predict, train_y))
    # print(class_report(test_predict, test_y))
    # plot_roc_auc(test_predict_proba, test_y)
    # plot_roc_auc(train_predict_proba, train_y)
    print(conf_matrix(test_predict, test_y))



def test_lgbm_forest():
    dataset = read_file_local('data_files/original_ds_normalized.csv')
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x, test_x, train_y, test_y = split_scikit(
        dataset[features], dataset['Y'], test_size=0.6
    )
    train_predict, test_predict, train_predict_proba, test_predict_proba = lgbm_forest(train_x, test_x, train_y, test_y)
    print(class_report(train_predict, train_y))
    print(class_report(test_predict, test_y))
    plot_roc_auc(test_predict_proba, test_y)
    plot_roc_auc(train_predict_proba, train_y)
    print(conf_matrix(test_predict, test_y))


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

# test_lin_regr()
# test_log_regr_rfe()
# test_decision_tree()
# test_random_forest()
test_xgb_forest()
# test_lgbm_forest()
#test_catboost_forest()
