import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# GitHub-hosted stable dataset URL (IBM mirror)
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

DATA_DIR = "data"
LOCAL_CSV_PATH = os.path.join(DATA_DIR, "telco_churn.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "telco_test_data.csv")


def save_test_dataset(X_test, y_test):
    """
    Save the test dataset (features + target) to /data folder.
    This dataset is used for Streamlit evaluation.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    test_df = X_test.copy()
    test_df["Churn"] = y_test.values

    test_df.to_csv(TEST_CSV_PATH, index=False)
    print(f"Test dataset saved at: {TEST_CSV_PATH}")


def load_data():
    """
    Load Telco Customer Churn dataset.
    - Downloads from IBM GitHub mirror if local CSV not found
    - Saves dataset locally for future use
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(LOCAL_CSV_PATH):
        print("Loaded dataset from local CSV.")
        df = pd.read_csv(LOCAL_CSV_PATH)
    else:
        print("Downloading dataset from GitHub mirror...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(LOCAL_CSV_PATH, index=False)
        print("Dataset saved locally at data/telco_churn.csv")

    return df



def clean_data(df):
    """
    Clean dataset:
    - Drop identifier
    - Fix data types
    - Handle missing values
    - Encode target
    """
    # Drop customerID (no predictive value)
    df = df.drop(columns=["customerID"], errors="ignore")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target variable
    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Reset index to avoid misalignment
    df = df.reset_index(drop=True)

    return df


def split_features_target(df):
    """
    Separate features and target
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def get_feature_types(X):
    """
    Identify categorical and numerical features
    """
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_features, numerical_features


def build_preprocessor(categorical_features, numerical_features):
    """
    Build preprocessing pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def split_data(X, y, random_state=42):
    """
    Split data into Train (70%), Validation (15%), Test (15%)
    """
    # Train + temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y
    )

    # Validation + test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Testing for data conversions
# df= clean_data(load_data())
# print(df.dtypes)