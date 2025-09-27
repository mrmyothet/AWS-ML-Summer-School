import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load
import os
from dotenv import load_dotenv
import boto3

# Load variables from a .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("REGION_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME,
)

buckets = s3.list_buckets()
print("Existing buckets:")
for bucket in buckets["Buckets"]:
    print(f'  {bucket["Name"]}')

bucket_name = "aws-ml-summer-school-043841769286"
file_key = "iris-xgb/iris.csv"

# df = pd.read_csv("./data/from_s3/iris.csv")

# s3.download_file(bucket_name, file_key, "./data/from_s3/iris.csv")
response = s3.get_object(Bucket=bucket_name, Key=file_key)
status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
if status == 200:
    print(f"Successful S3 get_object response. Status - {status}")
    df = pd.read_csv(response.get("Body"))

print(df.head())

df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# set features and target
X, y = df.iloc[:, :-1], df.iloc[:, -1]

print(X.shape)
print(y.shape)
print(y[:10])
print(X[:10])

x_train, y_train = X[:100], y[:100]
x_test, y_test = X[100:], y[100:]

print("X train : ", x_train.shape)
print("x test  : ", x_test.shape)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
y_pred

accuracy = np.sum(y_test == y_pred) / len(y_test)
print("Model Accuracy: ", accuracy)

# Save the model
dump(gnb, "group_1_project.joblib")
s3.upload_file(
    "group_1_project.joblib", bucket_name, "iris-xgb/model/group_1_project.joblib"
)
print("Model uploaded to S3")
