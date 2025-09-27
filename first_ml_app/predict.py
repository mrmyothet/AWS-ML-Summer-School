import json
import boto3
from joblib import load
import numpy as np


def handler(event, context):
    try:
        body = event.get("body")
        if body:
            body = json.loads(body)
            print(body)
            user_input = body.get("input", "")
        else:
            user_input = "Invalid User Input..."

        result = "Starting inferencing..."

        s3 = boto3.client("s3")

        bucket_name = "aws-ml-summer-school-043841769286"
        file_key = "iris-xgb/model/group_1_project.joblib"

        s3.download_file(bucket_name, file_key, "/tmp/group_1_project.joblib")

        loaded_model = load("/tmp/group_1_project.joblib")

        features = np.array(user_input).reshape(1, -1)
        prediction = loaded_model.predict(features)
        result = (
            prediction[0].item() if isinstance(prediction, np.ndarray) else prediction
        )

        return {"statusCode": 200, "body": f"Result: {result}"}

    except Exception as e:
        return {"statusCode": 400, "body": f"Error: {e}"}


# lambda_handler({"body": '{"input": [5.1, 3.5, 1.4, 0.2]}'}, "")
