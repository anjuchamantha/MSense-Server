from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import ML.RandomForest as rf

app = FastAPI()


class TestData(BaseModel):
    acc: str
    datetime: Optional[str] = None
    bat: str


@app.get("/")
async def root():
    return {"message": "Welcome to MSense-Server",
            "API Documentation": "/docs#/",
            "API Specification": "/redoc#/"}


@app.post("/test/{uid}")
async def test(uid: str, test_data: TestData):
    msg = "Test data received of the user: %s" % uid
    return {"message": msg,
            "test_data": test_data}


class PredictionData(BaseModel):
    screen_on_count: float
    screen_off_count: float
    battery_level: float
    charging_true_count: float
    charging_false_count: float
    charging_ac: float
    charging_usb: float
    charging_unknown: float


@app.post("/predict/")
def test(prediction_data: PredictionData):
    msg = "Prediction data received"
    data = [prediction_data.screen_on_count, prediction_data.screen_off_count, prediction_data.battery_level,
            prediction_data.charging_true_count, prediction_data.charging_false_count, prediction_data.charging_ac,
            prediction_data.charging_usb, prediction_data.charging_unknown]
    prediction = rf.rf_predict([data])
    print("Prediction:" , prediction)
    return {"message": msg,
            "request": prediction_data,
            "prediction": str(prediction)
            }
