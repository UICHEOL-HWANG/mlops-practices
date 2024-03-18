from fastapi.middleware.cors import CORSMiddleware
import mlflow
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import mlflow.pytorch
import torch
import torch.nn
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
import os 
import base64
from schemas import PredictIn,PredictOut

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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = image.resize((28, 28))
    image_np = np.array(image)
    image = torch.tensor(image_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    # 모델로부터 예측 결과를 얻습니다.
    with torch.no_grad():
        prediction = model(image).softmax(1).numpy()

    # 예측 결과를 시각화합니다.
    fig, ax = plt.subplots()
    ax.imshow(image_np, cmap='gray', interpolation='none')
    ax.set_title(f'Predict Number is {np.argmax(prediction)}')
    # 차트를 이미지 버퍼로 변환합니다.
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    # 예측 결과를 JSON으로 반환합니다.
    return JSONResponse(content={"prediction": np.argmax(prediction), "chart": img_str})


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