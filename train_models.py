from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def split_scikit(X, y, test_size=0.5, shuffle=True):
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


def split_dataset(dataset):
    pivot_index = int(len(dataset) / 3)
    train_ds, test_ds = dataset.head(pivot_index), dataset.tail(pivot_index)
    features = [f for f in dataset.columns.tolist() if not f == "Y"]
    train_x = train_ds[features]
    train_y = train_ds['Y']

    test_x = test_ds[features]
    test_y = test_ds['Y']

    return (train_x, train_y), (test_x, test_y)


def threshold(values, threshold_value=0.8):
    for i in range(0, len(values) + 1):
        if values[i - 1] >= threshold_value:
            values[i - 1] = 1
        else:
            values[i - 1] = 0
    return values


def linear_regression(train_x, test_x, train_y, test_y):
    lr = LinearRegression().fit(train_x, train_y)

    train_predict = threshold(lr.predict(train_x))
    test_predict = threshold(lr.predict(test_x))
    return train_predict, test_predict


def linear_regression_rfe(train_x, test_x, train_y, test_y, no_of_features=5):
    # select the most informative features
    rfe = RFE(LinearRegression(), no_of_features)
    selector = rfe.fit(train_x, train_y)
    train_predict = threshold(selector.predict(train_x))
    test_predict = threshold(selector.predict(test_x))
    return train_predict, test_predict


def logistic_regression(train_x, test_x, train_y, test_y):
    model = LogisticRegression().fit(train_x, train_y)
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    train_predict_proba = model.predict_proba(train_x)[::,1]
    test_predict_proba = model.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba


def logistic_regression_rfe(train_x, test_x, train_y, test_y, no_of_features=5):
    rfe = RFE(LogisticRegression(), no_of_features)
    selector = rfe.fit(train_x, train_y)
    train_predict = selector.predict(train_x)
    test_predict = selector.predict(test_x)

    train_predict_proba = selector.predict_proba(train_x)[::,1]
    test_predict_proba = selector.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba


def decision_tree(train_x, test_x, train_y, test_y, tree_depth=5):
    clf = tree.DecisionTreeClassifier(max_depth=tree_depth)
    clf = clf.fit(train_x, train_y)
    train_predict = clf.predict(train_x)
    test_predict = clf.predict(test_x)

    train_predict_proba = clf.predict_proba(train_x)[::,1]
    test_predict_proba = clf.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba


def random_forest(train_x, test_x, train_y, test_y, tree_depth=7, bootstrap=True):
    model_rf = RandomForestClassifier(n_estimators=200, max_depth=tree_depth, bootstrap=bootstrap, random_state=17)
    model_rf.fit(train_x, train_y)
    train_predict = model_rf.predict(train_x)
    test_predict = model_rf.predict(test_x)

    train_predict_proba = model_rf.predict_proba(train_x)[::,1]
    test_predict_proba = model_rf.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba


def xgboost_forest(train_x, test_x, train_y, test_y, tree_depth=7, bootstrap=True):
    model_rf = XGBClassifier(n_estimators=200, max_depth=tree_depth, bootstrap=bootstrap, random_state=17)
    model_rf.fit(train_x, train_y)
    train_predict = model_rf.predict(train_x)
    test_predict = model_rf.predict(test_x)

    train_predict_proba = model_rf.predict_proba(train_x)[::,1]
    test_predict_proba = model_rf.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba


def catboost_forest(train_x, test_x, train_y, test_y, tree_depth=7):
    model_rf = CatBoostClassifier(n_estimators=200, max_depth=tree_depth, random_state=17)
    model_rf.fit(train_x, train_y)
    train_predict = model_rf.predict(train_x)
    test_predict = model_rf.predict(test_x)
    train_predict_proba = model_rf.predict_proba(train_x)[::,1]
    test_predict_proba = model_rf.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba


def lgbm_forest(train_x, test_x, train_y, test_y, tree_depth=7, num_leaves=10):
    model_rf = LGBMClassifier(n_estimators=200, max_depth=tree_depth, num_leaves=num_leaves, random_state=17)
    model_rf.fit(train_x, train_y)
    train_predict = model_rf.predict(train_x)
    test_predict = model_rf.predict(test_x)
    train_predict_proba = model_rf.predict_proba(train_x)[::,1]
    test_predict_proba = model_rf.predict_proba(test_x)[::,1]
    return train_predict, test_predict, train_predict_proba, test_predict_proba

