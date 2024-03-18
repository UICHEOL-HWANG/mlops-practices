from fastapi.middleware.cors import CORSMiddleware
import mlflow
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import mlflow.pytorch
import torch
from PIL import Image
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    prediction = model(image).argmax(1).item()
    
    # 이미지를 Base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse(content={"prediction": prediction, "image": img_str})
