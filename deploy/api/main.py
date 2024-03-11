import mlflow
import pandas as pd 
from fastapi import FastAPI
from schemas import PredictIn, PredictOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from router.data import data_serve
from fastapi.middleware.cors import CORSMiddleware


# Create a FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}