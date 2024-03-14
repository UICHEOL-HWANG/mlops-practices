# 베이스 이미지 선택
FROM python:3.8-slim

# 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y git
RUN pip install torch torchvision mlflow psycopg2-binary boto3 

# 스크립트 추가
COPY . /app/mlflow-registry
WORKDIR /app/mlflow-registry

# 스크립트 실행
CMD ["python", "save_model_registry.py"]
