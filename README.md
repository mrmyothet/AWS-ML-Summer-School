# AWS-ML-Summer-School

### Connect to EC2 and Setup

```bash
ssh -i ~/Downloads/aws-ml-summer-school.pem ec2-user@18.136.194.60

chmod 0700 ~/Downloads/aws-ml-summer-school.pem

sudo yum update -y
sudo yum install -y python3 python3-pip

pip3 install numpy pandas sklearn matplotlib --user
```

### EC2 access to S3

```bash
aws --version
aws configure

cat ~/.aws/credentials

aws s3 ls s3://aws-ml-summer-school-043841769286/iris-xgb/data/

```

### AWS CLI

```bash
aws ec2 run-instances \
  --image-id ami-0c3fd0f5d33134a76 \
  --instance-type t2.micro \
  --key-name aws-ml-summer-school \
  --security-groups launch-wizard-1 \
  --count 1 \
  --region ap-southeast-1
```

### Push docker image to Amazon ECR

```bash
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 043841769286.dkr.ecr.ap-southeast-1.amazonaws.com

docker build -t first-ml-app-container .
docker buildx build --platform linux/amd64 -t first-ml-app-container:latest .

docker tag first-ml-app-container:latest 043841769286.dkr.ecr.ap-southeast-1.amazonaws.com/first-ml-app-container:latest

docker push 043841769286.dkr.ecr.ap-southeast-1.amazonaws.com/first-ml-app-container:latest
```

### Reference Links

- https://s3.us-east-1.amazonaws.com/media.sundog-soft.com/MLA-C01/AWS-Certified-ML-Engineer-Associate-Slides.pdf
- https://www.examtopics.com/exams/amazon/
