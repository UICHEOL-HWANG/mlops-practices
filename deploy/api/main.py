from fastapi.middleware.cors import CORSMiddleware
import mlflow
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import mlflow.pytorch
from IPython.display import HTML, Image
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
import os 
import base64

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://34.64.112.82:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://34.47.75.98:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# Create a FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = mlflow.pytorch.load_model("runs:/508bfe3947f6400f8fd6d6b68634b33c/model")
model.eval()


# 예측 엔드포인트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 파일 내용을 읽음
    contents = await file.read()
    # 이미지로 변환하고 그레이스케일로 처리
    image = Image.open(io.BytesIO(contents)).convert('L')
    # 28x28로 리사이즈
    image = image.resize((28, 28))
    # NumPy 배열로 변환하고 정규화
    image_np = np.array(image) / 255.0
    # 모델 입력을 위해 차원 추가
    tensor_image = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float()

    # 모델로 예측 수행
    with torch.no_grad():
        prediction = model(tensor_image)
        predicted_index = prediction.argmax(1).item()

    # 예측 결과와 사용자 이미지를 반환
    return JSONResponse(content={"prediction": predicted_index, "original_image": base64.b64encode(contents).decode('utf-8')})


@app.post("/accuracypredict", response_model=PredictOut)
async def accuracy_predict(predict_input: PredictIn):
    # 이미지 전처리
    image = Image.open(io.BytesIO(predict_input.image_data)).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)

    # 모델 인스턴스화 및 예측
    input_size = 784  # 28*28
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10  # 0-9 숫자
    model = Network(input_size, hidden_size1, hidden_size2, output_size)
    
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == predict_input.label).sum().item()
    accuracy = correct / len(image_tensor)

    return PredictOut(accuracy=accuracy)