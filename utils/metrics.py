import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def calculate_metrics(y_true, y_pred, threshold=0.5):
    auc, auc_per_class = compute_AUC(y_true, y_pred)
    y_pred = np.array(y_pred >= threshold, dtype=int)
    y_true = y_true.astype(int)
    TPR = compute_TPR(y_true, y_pred)

    metrics = {
        'AUC': auc,
        'TPR': TPR,
    }

    return metrics


def compute_AUC(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    auc_list = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_list.append(auc)

    return auc, auc_list


def compute_TPR(y_true, y_pred):
    # samples recall
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1

    return sum / count
