import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
    else:
        auc = None

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }

    return metrics


def get_confusion_matrix(model, X, y):
    """
    Return confusion matrix as a DataFrame
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    return pd.DataFrame(
        cm,
        index=["Actual_No", "Actual_Yes"],
        columns=["Pred_No", "Pred_Yes"]
    )


def get_classification_report(model, X, y):
    """
    Return classification report as a dictionary
    """
    y_pred = model.predict(X)
    return classification_report(y, y_pred, output_dict=True)
