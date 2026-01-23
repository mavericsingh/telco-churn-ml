import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    get_feature_types,
    build_preprocessor
)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_pickle(obj, filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


def main():
    # 1. Load & clean data
    df = load_data()
    df = clean_data(df)

    # 2. Split features / target
    X, y = split_features_target(df)

    # 3. Feature types
    categorical_features, numerical_features = get_feature_types(X)

    # 4. Shared preprocessor
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # Save preprocessor separately
    save_pickle(preprocessor, "preprocessor.pkl")

    # ================= Logistic Regression =================
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ))
    ])
    lr_pipeline.fit(X, y)
    save_pickle(lr_pipeline, "logistic_regression.pkl")

    # ================= Decision Tree (GridSearch best) =================
    dt_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=9,
            min_samples_leaf=50,
            min_samples_split=2,
            random_state=42
        ))
    ])
    dt_pipeline.fit(X, y)
    save_pickle(dt_pipeline, "decision_tree.pkl")

    # ================= kNN (GridSearch best) =================
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=21,
            weights="uniform"
        ))
    ])
    knn_pipeline.fit(X, y)
    save_pickle(knn_pipeline, "knn.pkl")

    # ================= Naive Bayes =================
    nb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])
    nb_pipeline.fit(X, y)
    save_pickle(nb_pipeline, "naive_bayes.pkl")

    # ================= Random Forest (GridSearch best) =================
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=10,
            max_features="log2",
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X, y)
    save_pickle(rf_pipeline, "random_forest.pkl")

    # ================= XGBoost (GridSearch best) =================
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
    xgb_pipeline.fit(X, y)
    save_pickle(xgb_pipeline, "xgboost.pkl")


if __name__ == "__main__":
    main()
