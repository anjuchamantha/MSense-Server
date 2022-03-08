from typing import Optional, List
from fastapi import FastAPI, Depends

from my_models.SensedData import SensedData
import ML.RandomForest as rf
from ML.tools import append_to_user_dataset, is_user_model_available, append_to_db_dataset
from sqlalchemy.orm import Session

import crud
import models
import schemas
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    return {"message": "Welcome to MSense-Server",
            "API Documentation": "/docs#/",
            "API Specification": "/redoc#/"}


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


@app.post("/predict_with_csv/{user_id}")
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
        append_to_user_dataset(user_id=user_id, sensed_data=sensed_data, meal_taken=float(meal_taken))

        return {"message": msg,
                "sensed data": sensed_data,
                "user id": user_id,
                "meal taken": meal_taken
                }
    else:
        # If the user's number of data points is >10, test with the received data and send the prediction
        if is_user_model_available(user_id=user_id):
            msg = "Prediction successful"
            prediction = rf.rf_predict([data], filename="ML/Saved Models/User Models/" + user_id)
        else:

            msg = "Not enough user data to train PERS. Tested with BASE."
            prediction = rf.rf_predict([data])

        print("Prediction:", prediction)
        return {"message": msg,
                "sensed data": sensed_data,
                "user id": user_id,
                "prediction": str(prediction)
                }


# ------------------------
@app.get("/db/dataset", response_model=List[schemas.Dataset])
def get_dataset(db: Session = Depends(get_db)):
    items = crud.get_full_data(db)
    return items


@app.post("/db/dataset", response_model=schemas.Dataset)
def create_dataset(row: schemas.DatasetCreate, db: Session = Depends(get_db)):
    return crud.create_eating_event(db=db, row=row)


@app.post("/predict/{user_id}")
async def predict_pers(user_id: str, sensed_data: schemas.DatasetCreate, meal_taken: Optional[str] = None,
                       db: Session = Depends(get_db)):
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
        append_to_db_dataset(user_id=user_id, sensed_data=sensed_data, meal_taken=float(meal_taken), db=db)

        return {"message": msg,
                "sensed data": sensed_data,
                "user id": user_id,
                "meal taken": meal_taken
                }
    else:
        # If the user's number of data points is >10, test with the received data and send the prediction
        if is_user_model_available(user_id=user_id):
            msg = "Prediction successful"
            prediction = rf.rf_predict([data], filename="ML/Saved Models/User Models/" + user_id)
        else:

            msg = "Not enough user data to train PERS. Tested with BASE."
            prediction = rf.rf_predict([data])

        print("Prediction:", prediction)
        return {"message": msg,
                "sensed data": sensed_data,
                "user id": user_id,
                "prediction": str(prediction)
                }

# ------------------------

# @app.get("/db/items", response_model=List[schemas.Item])
# def read_items(db: Session = Depends(get_db)):
#     items = crud.get_items(db)
#     return items
#
#
# @app.post("/db/items", response_model=schemas.Item)
# def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
#     return crud.create_user_item(db=db, item=item)
