import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
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


def train_knn_raw():
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

    # 6. Build RAW kNN pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform"
        ))
    ])

    # 7. Train model
    model.fit(X_train, y_train)

    # 8. Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    return model, train_metrics, val_metrics, test_metrics


def train_knn_gridsearch():
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
        ("classifier", KNeighborsClassifier())
    ])

    # 7. Hyperparameter grid
    param_grid = {
        "classifier__n_neighbors": [3, 5, 7, 11, 15, 21],
        "classifier__weights": ["uniform", "distance"]
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

    # 9. Fit grid search (training data only)
    grid_search.fit(X_train, y_train)

    # 10. Best model
    best_model = grid_search.best_estimator_

    # 11. Evaluate
    train_metrics = evaluate_model(best_model, X_train, y_train)
    val_metrics = evaluate_model(best_model, X_val, y_val)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    return best_model, grid_search.best_params_, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    # Overfitting RAW KNN classifier
    print("\n================ kNN (Raw Baseline) ================\n")

    _, train_metrics, val_metrics, test_metrics = train_knn_raw()

    print("kNN (Raw) - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nkNN (Raw) - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nkNN (Raw) - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n================ kNN (GridSearchCV) ================\n")

    best_model, best_params, train_metrics, val_metrics, test_metrics = (
        train_knn_gridsearch()
    )

    print("Best Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print("\nkNN - Training Metrics")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nkNN - Validation Metrics")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nkNN - Test Metrics")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
