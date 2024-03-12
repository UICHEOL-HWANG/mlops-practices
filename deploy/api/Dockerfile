# 이미지 빌드를 위한 기본 이미지 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /usr/api_serve

# 현재 디렉토리의 파일들을 컨테이너의 작업 디렉토리에 복사
COPY . /usr/api_serve/

# 필요한 Python 패키지 설치
RUN pip install -U pip &&\
    pip install boto3 mlflow "fastapi[all]" pandas scikit-learn==1.3.2 psycopg2-binary

# FastAPI 애플리케이션 실행 명령 설정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]