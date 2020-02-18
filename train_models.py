from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, cross_val_score


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
    train_predict = threshold(model.predict(train_x))
    test_predict = threshold(model.predict(test_x))
    return train_predict, test_predict


def logistic_regression_rfe(train_x, test_x, train_y, test_y, no_of_features=5):
    rfe = RFE(LogisticRegression(), no_of_features)
    selector = rfe.fit(train_x, train_y)
    train_predict = threshold(selector.predict(train_x))
    test_predict = threshold(selector.predict(test_x))
    return train_predict, test_predict


def decision_tree(train_x, test_x, train_y, test_y, tree_depth=5):
    clf = tree.DecisionTreeClassifier(max_depth=tree_depth)
    clf = clf.fit(train_x, train_y)
    train_predict = threshold(clf.predict(train_x))
    test_predict = threshold(clf.predict(test_x))
    return train_predict, test_predict


def random_forest(train_x, test_x, train_y, test_y, tree_depth=5):
    model_rf = RandomForestClassifier(n_estimators=100, n_jobs=4,
                                      max_depth=None, random_state=17)
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    cv = ShuffleSplit(n_splits=n_fold, test_size=0.3,
                      random_state=17)
    # calcuate ROC-AUC for each split
    cv_scores_rf = cross_val_score(model_rf, train_x, train_y, cv=cv, scoring='roc_auc')
    print(cv_scores_rf)
    model_rf.fit(train_x, train_y)
    y_scores = model_rf.predict(test_x)
    print(classification_report(test_y.values, y_scores))

