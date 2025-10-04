dockerfile_content = '''
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    xgboost==1.7.3 \\
    pandas==1.5.3 \\
    scikit-learn==1.2.2 \\
    boto3==1.26.137 \\
    sagemaker-training==4.4.3 \\
    joblib==1.2.0 \\
    numpy==1.24.3

# Copy training code
COPY train_custom.py /opt/ml/code/train_custom.py

# Set environment variables
ENV SAGEMAKER_PROGRAM train_custom.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

WORKDIR /opt/ml/code