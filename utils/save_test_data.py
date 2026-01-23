from utils.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    split_data,
    save_test_dataset
)

def main():
    df = clean_data(load_data())
    X, y = split_features_target(df)
    _, _, X_test, _, _, y_test = split_data(X, y)
    save_test_dataset(X_test, y_test)

if __name__ == "__main__":
    main()
