import argparse
import json
import logging
import os
import pandas as pd
import xgboost as xgb
import joblib
from io import StringIO
import boto3
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, content_type='text/csv'):
    """Parse input data for inference"""
    if content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    return model.predict(input_data)

def output_fn(prediction, accept='text/csv'):
    """Format prediction output"""
    if accept == 'text/csv':
        return pd.DataFrame(prediction, columns=['prediction']).to_csv(index=False)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Custom hyperparameters
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.3)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--custom-preprocessing', type=str, default='standard')
    parser.add_argument('--feature-engineering', type=str, default='basic')
    
    # SageMaker arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()
    
    logger.info(f"Loading training data from {args.train}")
    
    # Custom data loading logic
    train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
    logger.info(f"Found training files: {train_files}")
    
    train_data = pd.concat([
        pd.read_csv(os.path.join(args.train, file)) 
        for file in train_files
    ])
    
    if args.validation:
        val_files = [f for f in os.listdir(args.validation) if f.endswith('.csv')]
        logger.info(f"Found validation files: {val_files}")
        val_data = pd.concat([
            pd.read_csv(os.path.join(args.validation, file))
            for file in val_files
        ])
    
    logger.info(f"Train data shape: {train_data.shape}")
    if args.validation:
        logger.info(f"Validation data shape: {val_data.shape}")
    
    # Custom preprocessing based on hyperparameter
    if args.custom_preprocessing == 'advanced':
        logger.info("Applying advanced preprocessing...")
        # Custom feature engineering
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('target')
        
        # Add feature interactions for first two numeric features
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            train_data[f'{col1}_{col2}_interaction'] = train_data[col1] * train_data[col2]
            if args.validation:
                val_data[f'{col1}_{col2}_interaction'] = val_data[col1] * val_data[col2]
        
        # Fill missing values with median
        train_data = train_data.fillna(train_data.median())
        if args.validation:
            val_data = val_data.fillna(val_data.median())
    
    # Feature engineering
    if args.feature_engineering == 'advanced':
        logger.info("Applying advanced feature engineering...")
        # Create polynomial features for first numeric column
        first_numeric = train_data.select_dtypes(include=[np.number]).columns[0]
        train_data[f'{first_numeric}_squared'] = train_data[first_numeric] ** 2
        if args.validation:
            val_data[f'{first_numeric}_squared'] = val_data[first_numeric] ** 2
    
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    
    if args.validation:
        X_val = val_data.drop('target', axis=1)
        y_val = val_data['target']
    
    logger.info(f"Final feature count: {X_train.shape[1]}")
    
    # Custom XGBoost configuration with advanced parameters
    model = xgb.XGBRegressor(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        tree_method='hist',  # Custom parameter for faster training
        enable_categorical=False,  # Custom parameter
        random_state=42,
        eval_metric='rmse'
    )
    
    logger.info("Training model...")
    
    # Train with early stopping if validation data available
    if args.validation:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_rmse = mean_squared_error(y_train, train_pred, squared=False)
        val_rmse = mean_squared_error(y_val, val_pred, squared=False)
        
        logger.info(f"Training RMSE: {train_rmse:.4f}")
        logger.info(f"Validation RMSE: {val_rmse:.4f}")
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        train_rmse = mean_squared_error(y_train, train_pred, squared=False)
        logger.info(f"Training RMSE: {train_rmse:.4f}")
    
    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names for inference
    feature_names_path = os.path.join(args.model_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(list(X_train.columns), f)
    logger.info(f"Feature names saved to {feature_names_path}")