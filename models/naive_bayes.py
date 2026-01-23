import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

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
    split_data
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def build_nb_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_features),
        ]
    )
    return preprocessor


def train_naive_bayes_raw():
    # 1. Load & clean data
    df = load_data()
    df = clean_data(df)

    # 2. Split features & target
    X, y = split_features_target(df)

    # 3. Build Naive Bayes specific preprocessor
    preprocessor = build_nb_preprocessor(X)

    # 4. Train / Validation / Test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Build pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    return model, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":

    print("\n================ Naive Bayes (Raw Baseline) ================\n")

    _, train_metrics, val_metrics, test_metrics = train_naive_bayes_raw()

    print("Naive Bayes - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nNaive Bayes - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nNaive Bayes - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
