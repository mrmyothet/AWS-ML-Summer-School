# AWS-ML-Summer-School

### Connect to EC2 and Setup

```bash
ssh -i ~/Downloads/aws-ml-summer-school.pem ec2-user@18.136.194.60

chmod 0700 ~/Downloads/aws-ml-summer-school.pem

sudo yum update -y
sudo yum install -y python3 python3-pip

pip3 install numpy pandas scikit-learn matplotlib --user
```
