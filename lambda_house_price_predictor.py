import json
import math


def lambda_handler(event, context):
    """
    Simple house price predictor using basic linear model
    Input: {bedrooms, bathrooms, sqft, location_score}
    Output: {predicted_price, confidence}
    """
    print(" Starting house price prediction...")
    try:
        # Parse input - handle both API Gateway and direct calls
        if "body" in event:
            # Called via API Gateway
            input_data = json.loads(event["body"])
        else:

            # Direct Lambda invocation
            input_data = event
            # Extract features
            bedrooms = input_data.get("bedrooms", 3)
            bathrooms = input_data.get("bathrooms", 2)
            sqft = input_data.get("sqft", 1500)
            location_score = input_data.get("location_score", 5)  # 1-10 scale
            print(
                f"Input: {bedrooms}BR, {bathrooms}BA, {sqft}sqft, location: {location_score}"
            )

            # Simple price prediction model (placeholder for real model)
            base_price = 50000  # Base price
            price_per_sqft = 100
            bedroom_value = 15000
            bathroom_value = 10000
            location_multiplier = location_score / 10
            predicted_price = (
                base_price
                + (sqft * price_per_sqft)
                + (bedrooms * bedroom_value)
                + (bathrooms * bathroom_value)
            ) * (1 + location_multiplier)

            # Calculate confidence (simplified)
            confidence = min(95, max(60, 85 - abs(sqft - 2000) / 100))
            # Prepare response
            result = {
                "predicted_price": round(predicted_price, 2),
                "confidence_percent": round(confidence, 1),
                "input_features": {
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "sqft": sqft,
                    "location_score": location_score,
                },
                "model_version": "1.0",
                "prediction_timestamp": context.aws_request_id,
            }
            print(
                f"Prediction: ${predicted_price:, .2f} (confidence: {confidence:.1f}%)"
            )
            # Return response (format depends on how Lambda is called)

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(result),
            }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


# Event Json
# {"bedrooms": 3, "bathrooms": 2, "sqft": 1900, "location_score": 7}
