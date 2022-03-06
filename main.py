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


class SensedData(BaseModel):
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
    charging_ac: float
    charging_usb: float
    charging_unknown: float

    minutes_elapsed: float
    hours_elapsed: float
    weekend: float
    radius_of_gyration: float

    screen_on_count: float
    screen_off_count: float


@app.post("/predict")
async def predict(sensed_data: SensedData):
    msg = "Prediction data received"
    data = [sensed_data.acc_x, sensed_data.acc_y, sensed_data.acc_z, sensed_data.acc_x_bef,
            sensed_data.acc_y_bef, sensed_data.acc_z_bef, sensed_data.acc_x_aft, sensed_data.acc_y_aft,
            sensed_data.acc_z_aft,
            sensed_data.acc_xabs, sensed_data.acc_yabs, sensed_data.acc_zabs, sensed_data.acc_xabs_bef,
            sensed_data.acc_yabs_bef, sensed_data.acc_zabs_bef,
            sensed_data.acc_xabs_aft,
            sensed_data.acc_yabs_aft, sensed_data.acc_zabs_aft, sensed_data.battery_level,
            sensed_data.charging_true_count,
            sensed_data.charging_ac, sensed_data.charging_usb,
            sensed_data.charging_unknown, sensed_data.minutes_elapsed, sensed_data.hours_elapsed,
            sensed_data.weekend, sensed_data.radius_of_gyration,
            sensed_data.screen_on_count, sensed_data.screen_off_count]
    prediction = rf.rf_predict([data])
    print("Prediction:", prediction)
    return {"message": msg,
            "sensed data": sensed_data,
            "prediction": str(prediction)
            }


@app.post("/predict/{user_id}")
async def predict_pers(user_id: str, sensed_data: SensedData, meal_taken: Optional[str] = None):
    data = [sensed_data.acc_x, sensed_data.acc_y, sensed_data.acc_z, sensed_data.acc_x_bef,
            sensed_data.acc_y_bef, sensed_data.acc_z_bef, sensed_data.acc_x_aft, sensed_data.acc_y_aft,
            sensed_data.acc_z_aft,
            sensed_data.acc_xabs, sensed_data.acc_yabs, sensed_data.acc_zabs, sensed_data.acc_xabs_bef,
            sensed_data.acc_yabs_bef, sensed_data.acc_zabs_bef,
            sensed_data.acc_xabs_aft,
            sensed_data.acc_yabs_aft, sensed_data.acc_zabs_aft, sensed_data.battery_level,
            sensed_data.charging_true_count,
            sensed_data.charging_ac, sensed_data.charging_usb,
            sensed_data.charging_unknown, sensed_data.minutes_elapsed, sensed_data.hours_elapsed,
            sensed_data.weekend, sensed_data.radius_of_gyration,
            sensed_data.screen_on_count, sensed_data.screen_off_count]

    if meal_taken:
        # Append the new datapoint with the ground truth
        msg = "New data point received and saved with ground truth"

        return {"message": msg,
                "sensed data": sensed_data,
                "user id": user_id,
                "meal taken": meal_taken
                }
    else:
        # If the user's number of data points is >10, test with the received data and send the prediction
        msg = "Prediction successful"
        prediction = rf.rf_predict([data])

        print("Prediction:", prediction)
        return {"message": msg,
                "sensed data": sensed_data,
                "user id": user_id,
                "prediction": str(prediction)
                }


# @app.post("/train")
# async def train():
#     results = ML_main.train()
#     return {"message": "Trained and Saved .pkl Successfully",
#             "result": results
#             }
