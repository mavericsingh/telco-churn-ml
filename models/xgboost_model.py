import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

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


def train_xgboost_raw():
    # 1. Load & clean data
    df = load_data()
    df = clean_data(df)

    # 2. Split features & target
    X, y = split_features_target(df)

    # 3. Identify feature types
    categorical_features, numerical_features = get_feature_types(X)

    # 4. Preprocessor
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # 5. Train / Validation / Test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 6. Raw XGBoost model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 7. Train
    model.fit(X_train, y_train)

    # 8. Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    return model, train_metrics, val_metrics, test_metrics


def train_xgboost_gridsearch():
    # 1. Load & clean data
    df = load_data()
    df = clean_data(df)

    # 2. Split features & target
    X, y = split_features_target(df)

    # 3. Identify feature types
    categorical_features, numerical_features = get_feature_types(X)

    # 4. Preprocessor
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # 5. Train / Validation / Test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 6. Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 7. GridSearchCV
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # 8. Fit ONLY on training data
    grid_search.fit(X_train, y_train)

    # 9. Best model
    best_model = grid_search.best_estimator_

    # 10. Evaluate
    train_metrics = evaluate_model(best_model, X_train, y_train)
    val_metrics = evaluate_model(best_model, X_val, y_val)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    return best_model, grid_search.best_params_, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":

    # RAW XGboost leads to overfitting
    print("\n================ XGBoost (Raw Baseline) ================\n")

    _, train_metrics, val_metrics, test_metrics = train_xgboost_raw()

    print("XGBoost (Raw) - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nXGBoost (Raw) - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nXGBoost (Raw) - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n================ XGBoost (GridSearchCV) ================\n")

    best_model, best_params, train_metrics, val_metrics, test_metrics = (
        train_xgboost_gridsearch()
    )

    print("Best Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print("\nXGBoost (GridSearch) - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nXGBoost (GridSearch) - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nXGBoost (GridSearch) - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
