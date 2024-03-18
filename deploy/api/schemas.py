from pydantic import BaseModel

class PredictIn(BaseModel):
    image_data: bytes  
    label: int

class PredictOut(BaseModel):
    accuracy: float