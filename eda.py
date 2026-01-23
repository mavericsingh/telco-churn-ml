"""
Exploratory Data Analysis (EDA) for Telco Customer Churn Dataset

Purpose:
- Understand dataset structure
- Identify feature types
- Detect missing values
- Analyze target distribution

Note:
This file is for exploratory analysis only.
No data modification or model training is performed here.
"""

import pandas as pd
from utils.preprocessing import load_data
import seaborn as sns
import matplotlib.pyplot as plt


def run_eda():
    # Load dataset (from local CSV or UCI)
    df = load_data()

    print("\n================ DATASET OVERVIEW ================\n")

    # 1. Dataset shape
    print("Dataset Shape (Rows, Columns):")
    print(df.shape)

    # 2. Preview dataset
    print("\nFirst 5 Rows:")
    print(df.head())

    # 3. Dataset information
    print("\nDataset Info:")
    df.info()

    # 4. Statistical summary (numerical features)
    print("\nStatistical Summary (Numerical Features):")
    print(df.describe())

    # 5. Missing value analysis
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # 6. Target variable distribution
    print("\nTarget Variable Distribution (Churn):")
    print(df["Churn"].value_counts())

    print("\nTarget Variable Distribution (Percentage):")
    print(df["Churn"].value_counts(normalize=True) * 100)

    # Correlation analysis (numerical features only)
    numerical_df = df.select_dtypes(include=["int64", "float64"])

    print("\nCorrelation Matrix (Numerical Features):")
    print(numerical_df.corr())

    plt.figure(figsize=(6, 4))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.show()


    print("\n================ END OF EDA =================\n")


if __name__ == "__main__":
    run_eda()