import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters from estimator
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)

    args = parser.parse_args()

    # SageMaker channels
    train_path = os.path.join("/opt/ml/input/data/train", "train.csv")
    val_path = os.path.join("/opt/ml/input/data/validation", "validation.csv")

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )

    model.fit(X_train, y_train)

    # Save model to model_dir (SageMaker will copy it to S3)
    model_dir = os.environ["SM_MODEL_DIR"]
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
