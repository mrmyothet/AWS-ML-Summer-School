import json


def lambda_handler(event, context):
    # TODO implement

    try:
        body = event.get("body")
        if body:
            body = json.loads(body)
            print(body)
            user_input = body.get("input", "")
        else:
            user_input = "Invalid User Input..."

        result = "Starting inferencing..."
        return {"statusCode": 200, "body": f"Result: {result}"}

    except Exception as e:
        return {"statusCode": 400, "body": f"Error: {e}"}
