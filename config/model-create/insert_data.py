import psycopg2
from torchvision import datasets, transforms
import io
import torch

# PostgreSQL 연결 설정
conn = psycopg2.connect(
    dbname="mydatabase",
    user="myuser",
    password="mypassword",
    host="10.15.36.232",
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
    # 이미지를 BytesIO 객체로 변환
    buffer = io.BytesIO()
    # PIL 이미지로 변환
    pil_image = transforms.ToPILImage()(image)
    # PIL 이미지를 BytesIO 객체로 저장
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_data = buffer.read()

    # PostgreSQL에 이미지 및 레이블 삽입
    cur.execute("INSERT INTO mnist_images (image_data, label) VALUES (%s, %s)", (image_data, label))
    conn.commit()

# 연결 종료
cur.close()
conn.close()
