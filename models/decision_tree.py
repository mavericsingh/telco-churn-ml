import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

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
    plt.title("Learning Curve - Decision Tree")
    plt.legend()
    plt.grid(True)

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/decision_tree_accuracy_curve.png", bbox_inches="tight")
    plt.close()


def train_decision_tree():
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
        ("classifier", DecisionTreeClassifier(
            random_state=42
        ))
    ])

    # 7. Train model
    model.fit(X_train, y_train)

    # 8. Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    # # 9. Learning curve (CV-based, global data)
    # plot_learning_curve(model, X, y)

    return model, train_metrics, val_metrics, test_metrics


def train_decision_tree_gridsearch():
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
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

    # 7. Parameter grid
    param_grid = {
        "classifier__max_depth": [3, 5, 7, 9, None],
        "classifier__min_samples_leaf": [10, 20, 50, 100],
        "classifier__min_samples_split": [2, 10, 20]
    }

    # 8. GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # 9. Fit grid search (ONLY on training data)
    grid_search.fit(X_train, y_train)

    # 10. Best model
    best_model = grid_search.best_estimator_

    # 11. Evaluate best model
    train_metrics = evaluate_model(best_model, X_train, y_train)
    val_metrics = evaluate_model(best_model, X_val, y_val)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    return best_model, grid_search.best_params_, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":

    print("\n================ Decision Tree (RAW) ================\n")
    # Raw DT inplementation leads to overfitting
    model, train_metrics, val_metrics, test_metrics = train_decision_tree()

    print("\nDecision Tree - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDecision Tree - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDecision Tree - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n================ Decision Tree (GridSearchCV) ================\n")

    best_model, best_params, train_metrics, val_metrics, test_metrics = (
        train_decision_tree_gridsearch()
    )

    print("Best Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print("\nDecision Tree (GridSearch) - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDecision Tree (GridSearch) - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDecision Tree (GridSearch) - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
