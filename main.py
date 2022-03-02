from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import ML.RandomForest as rf

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to MSense-Server",
            "API Documentation": "/docs#/",
            "API Specification": "/redoc#/"}


class PredictionData(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    acc_x_bef: float
    acc_y_bef: float
    acc_z_bef: float
    acc_x_aft: float
    acc_y_aft: float
    acc_z_aft: float
    acc_xabs: float
    acc_yabs: float
    acc_zabs: float
    acc_xabs_bef: float
    acc_yabs_bef: float
    acc_zabs_bef: float
    acc_xabs_aft: float
    acc_yabs_aft: float
    acc_zabs_aft: float

    battery_level: float
    charging_true_count: float
    charging_false_count: float
    charging_ac: float
    charging_usb: float
    charging_unknown: float

    minutes_elapsed: float
    hours_elapsed: float
    weekend: float
    radius_of_gyration: float

    screen_on_count: float
    screen_off_count: float


@app.post("/predict/")
async def test(prediction_data: PredictionData):
    msg = "Prediction data received"
    data = [prediction_data.acc_x, prediction_data.acc_y, prediction_data.acc_z, prediction_data.acc_x_bef,
            prediction_data.acc_y_bef, prediction_data.acc_z_bef, prediction_data.acc_x_aft, prediction_data.acc_y_aft,
            prediction_data.acc_z_aft,
            prediction_data.acc_xabs, prediction_data.acc_yabs, prediction_data.acc_zabs, prediction_data.acc_xabs_bef,
            prediction_data.acc_yabs_bef, prediction_data.acc_zabs_bef,
            prediction_data.acc_xabs_aft,
            prediction_data.acc_yabs_aft, prediction_data.acc_zabs_aft, prediction_data.battery_level,
            prediction_data.charging_true_count, prediction_data.charging_false_count,
            prediction_data.charging_ac, prediction_data.charging_usb,
            prediction_data.charging_unknown, prediction_data.minutes_elapsed, prediction_data.hours_elapsed,
            prediction_data.weekend, prediction_data.radius_of_gyration,
            prediction_data.screen_on_count, prediction_data.screen_off_count]
    prediction = rf.rf_predict([data])
    print("Prediction:", prediction)
    return {"message": msg,
            "request": prediction_data,
            "prediction": str(prediction)
            }
