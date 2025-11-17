# train.py

import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump


def load_and_prepare_data(data_path: str):
    """
    Load the cardio dataset and do a small amount of cleaning
    before training the model.
    """
    # Kaggle cardio dataset uses ';' as separator
    df = pd.read_csv(data_path, sep=";")

    # Convert age from days to years
    df["age_years"] = (df["age"] / 365.25).round(1)
    df = df.drop(columns=["age"])

    # Filter out clearly unrealistic values (simple sanity checks)
    df = df[(df["height"] >= 120) & (df["height"] <= 220)]
    df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]

    # Keep only reasonable blood pressure ranges
    df = df[(df["ap_hi"].between(90, 250)) & (df["ap_lo"].between(40, 200))]

    # Drop rows with any remaining missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=["cardio", "id"])
    y = df["cardio"]

    return X, y


def main():
    # project_root = .../Projects/project
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "dataset" / "cardio_train.csv"

    print(f"Using dataset at: {data_path}")

    # Load and prepare data
    X, y = load_and_prepare_data(str(data_path))

    # Remember the feature order for prediction time
    feature_columns = X.columns.tolist()

    # Final model: gradient boosting
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    # Save model and feature list (will be saved in the *current* folder)
    dump(model, "cardio_model.joblib")
    dump(feature_columns, "cardio_feature_columns.joblib")

    print("Saved cardio_model.joblib and cardio_feature_columns.joblib")
    print(f"Trained on {len(X)} rows and {len(feature_columns)} features.")


if __name__ == "__main__":
    main()
