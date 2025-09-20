import os
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


import boto3
import pandas as pd
from io import StringIO

# Create S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-southeast-1",
)


# Read CSV directly into pandas (without downloading)
response = s3.get_object(
    Bucket="aws-ml-summer-school-043841769286", Key="iris-xgb/iris.csv"
)
df = pd.read_csv(response["Body"])
print("Data loaded directly into pandas:")
print(df.head())


# 1. Load data
iris = load_iris()  ###  LOAD FROM S3 using BOTO3


X = iris.data
y = iris.target

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Train model
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)


# 4. Predictions
y_pred = clf.predict(X_test)


# 5. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
