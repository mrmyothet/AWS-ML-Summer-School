import json
import boto3
import os

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket = os.environ.get("BUCKET_NAME", "aws-ml-summer-school-043841769286")
    prefix = os.environ.get("S3_PREFIX", "iris-xgb/data/")  # optional

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        return {
            "statusCode": 200,
            "body": json.dumps({"bucket": bucket, "prefix": prefix, "keys": keys})
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}