from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float
from sqlalchemy.orm import relationship

from database import Base


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    user_id = Column(String, index=True)
    meal_taken = Column(Float, index=True)
    prediction = Column(Float, index=True)

    timestamp = Column(String, index=True)

    acc_x = Column(Float)
    acc_y = Column(Float)
    acc_z = Column(Float)
    acc_x_bef = Column(Float)
    acc_y_bef = Column(Float)
    acc_z_bef = Column(Float)
    acc_x_aft = Column(Float)
    acc_y_aft = Column(Float)
    acc_z_aft = Column(Float)
    acc_xabs = Column(Float)
    acc_yabs = Column(Float)
    acc_zabs = Column(Float)
    acc_xabs_bef = Column(Float)
    acc_yabs_bef = Column(Float)
    acc_zabs_bef = Column(Float)
    acc_xabs_aft = Column(Float)
    acc_yabs_aft = Column(Float)
    acc_zabs_aft = Column(Float)

    battery_level = Column(Float)
    charging_true_count = Column(Float)
    charging_ac = Column(Float)
    charging_usb = Column(Float)
    charging_unknown = Column(Float)

    minutes_elapsed = Column(Float)
    hours_elapsed = Column(Float)
    weekend = Column(Float)
    radius_of_gyration = Column(Float)

    screen_on_count = Column(Float)
    screen_off_count = Column(Float)


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
