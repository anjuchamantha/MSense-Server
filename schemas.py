from typing import Optional

from pydantic import BaseModel


# create
class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None


class ItemCreate(ItemBase):
    pass


# read
class Item(ItemBase):
    id: int

    class Config:
        orm_mode = True


# ---------------------
class DatasetBase(BaseModel):
    # user_id: str
    # meal_taken: float
    timestamp: str

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


class DatasetCreate(DatasetBase):
    pass


class Dataset(DatasetBase):
    id: int

    class Config:
        orm_mode = True
