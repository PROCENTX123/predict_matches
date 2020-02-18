from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score


def class_report(y_predict, y_actual):
    return classification_report(y_actual, y_predict)

def conf_matrix(y_predict, y_actual):
    return confusion_matrix(y_actual, y_predict)

def lloss(y_predict, y_actual):
    return log_loss(y_actual, y_predict, eps=1e-15)

def roc_auc(y_predict, y_actual):
    return roc_auc_score(y_actual, y_predict)