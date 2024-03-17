import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import psycopg2
import mlflow
from torchvision import transforms
from PIL import Image
import io

# MLflow 설정
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://34.64.144.27:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://34.64.119.81:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 데이터베이스 연결
db_connect = psycopg2.connect(
    user="myuser",
    password="mypassword",
    host="10.15.36.232",
    port=5432,
    database="mydatabase",
)

# 신경망 정의
class Network(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.sigmoid(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output_layer(x)

# 데이터 로딩
class MNISTDataset(Dataset):
    def __init__(self, cursor, transform=None):
        self.cursor = cursor
        self.transform = transform

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM mnist_images")
        return self.cursor.fetchone()[0]

    def __getitem__(self, idx):
        self.cursor.execute("SELECT image_data, label FROM mnist_images LIMIT 1 OFFSET %s", (idx,))
        image_data, label = self.cursor.fetchone()
        # BytesIO 객체에서 PIL 이미지를 읽어옴
        image = Image.open(io.BytesIO(image_data)).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cursor = db_connect.cursor()
dataset = MNISTDataset(cursor, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 모델, 손실 함수, 옵티마이저 설정
input_size = 784  # 28*28
hidden_size1 = 128
hidden_size2 = 64
output_size = 10  # 0-9 숫자

model = Network(input_size, hidden_size1, hidden_size2, output_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# MLflow 로깅 시작
with mlflow.start_run():
    for epoch in range(1):  # 에포크 수
        total_loss = 0 # 사정상 에포크는 1번만
        total_correct = 0
        for images, labels in dataloader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = 100 * total_correct / len(dataloader.dataset)

        print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # 모델 저장
    mlflow.pytorch.log_model(model, "model")

# 데이터베이스 연결 종료
cursor.close()
db_connect.close()
