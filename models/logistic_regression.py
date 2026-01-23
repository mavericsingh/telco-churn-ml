import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from utils.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    get_feature_types,
    build_preprocessor,
    split_data
)


def evaluate_model(model, X, y):
    """
    Compute all required evaluation metrics
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    }

    return metrics


def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training Accuracy")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/logistic_regression_accuracy_curve.png", bbox_inches="tight")
    plt.close()


def plot_loss_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="neg_log_loss",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    plt.figure()
    plt.plot(train_sizes, -train_scores.mean(axis=1), marker="o", label="Training Loss")
    plt.plot(train_sizes, -val_scores.mean(axis=1), marker="o", label="Validation Loss")
    plt.xlabel("Training Set Size")
    plt.ylabel("Log Loss")
    plt.title("Loss Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/logistic_regression_loss_curve.png", bbox_inches="tight")
    plt.close()



def train_logistic_regression():
    # 1. Load & clean data
    df = load_data()
    df = clean_data(df)

    # 2. Split features & target
    X, y = split_features_target(df)

    # 3. Identify feature types
    categorical_features, numerical_features = get_feature_types(X)

    # 4. Build preprocessor
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # 5. Train / Validation / Test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 6. Build pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ))
    ])

    # 7. Train model
    model.fit(X_train, y_train)

    # 8. Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    # # 9. Evaluate model:
    # # Plot learning curve (Training vs Validation)
    # plot_learning_curve(model, X, y)

    # # Plot loss learning curve
    # plot_loss_curve(model, X, y)

    return model, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    model, train_metrics, val_metrics, test_metrics = train_logistic_regression()

    print("\nLogistic Regression - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nLogistic Regression - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nLogistic Regression - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

