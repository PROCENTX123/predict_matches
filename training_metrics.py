from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score, roc_curve


def class_report(y_predict, y_actual):
    return classification_report(y_actual, y_predict)


def conf_matrix(y_predict, y_actual):
    return confusion_matrix(y_actual, y_predict)


def lloss(y_predict, y_actual):
    return log_loss(y_actual, y_predict, eps=1e-15)


def roc_auc(y_predict, y_actual):
    return roc_auc_score(y_actual, y_predict)


def plot_roc_auc(predict_proba, true_proba):
    fpr, tpr, _ = roc_curve(true_proba, predict_proba)
    auc = roc_auc_score(true_proba, predict_proba)
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

