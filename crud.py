from sqlalchemy.orm import Session

import models
import schemas


# Read all the rows of a user
def get_user_rows(db: Session, user_id: str):
    print("Reading rows of user: ", user_id)
    return db.query(models.Dataset).filter(models.Dataset.user_id == user_id).all()


# add a row to the dataset table
def create_eating_event(db: Session, row: schemas.DatasetCreate, user_id: str, meal_taken: float, prediction: float):
    print("Adding a row of a user")
    db_item = models.Dataset(**row.dict(), user_id=user_id, meal_taken=meal_taken, prediction=prediction)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


# read all the data
def get_full_data(db: Session):
    return db.query(models.Dataset).all()


# -------------------- examples
# READ

def get_items(db: Session):
    return db.query(models.Item).all()


# Create
def create_user_item(db: Session, item: schemas.ItemCreate):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
