import psycopg2
from torchvision import datasets, transforms
import io
import torch

# PostgreSQL 연결 설정
conn = psycopg2.connect(
    dbname="mydatabase",
    user="myuser",
    password="mypassword",
    host="postgres-server",
    port=5432
)
cur = conn.cursor()

# MNIST 데이터셋 다운로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(torch.float32)),
])

mnist_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

# PostgreSQL에 데이터 삽입
for image, label in mnist_dataset:
    # 이미지 데이터를 바이트로 변환
    buffer = io.BytesIO()
    torch.save(image, buffer)
    buffer.seek(0)
    image_data = buffer.read()

    # PostgreSQL에 이미지 및 레이블 삽입
    cur.execute("INSERT INTO mnist_images (image_data, label) VALUES (%s, %s)", (image_data, label))
    conn.commit()

# 연결 종료
cur.close()
conn.close()
