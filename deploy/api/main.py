from fastapi.middleware.cors import CORSMiddleware
import mlflow
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import mlflow.pytorch
import torch
from PIL import Image
import io
import numpy as np

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://34.64.144.27:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://34.64.119.81:5000"
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


model = mlflow.pytorch.load_model("runs:/2fd50ae92cbd43efbe305a1d7f457596/model")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    prediction = model(image).argmax(1).item()
    return JSONResponse(content={"prediction": prediction})