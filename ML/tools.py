import os
import openpyxl
import csv
from models.SensedData import SensedData
import ML.BASE as BASE
import pandas as pd
from pathlib import Path


def feature_names_and_values(sensed_data):
    all_features_server = ["acc_x", "acc_y", "acc_z", "acc_x_bef", "acc_y_bef", "acc_z_bef", "acc_x_aft", "acc_y_aft",
                           "acc_z_aft",
                           "acc_xabs", "acc_yabs", "acc_zabs", "acc_xabs_bef", "acc_yabs_bef", "acc_zabs_bef",
                           "acc_xabs_aft",
                           "acc_yabs_aft", "acc_zabs_aft", "battery_level", "charging_true_count",
                           "charging_ac", "charging_usb",
                           "charging_unknown", "minutes_elapsed", "hours_elapsed", "weekend", "radius_of_gyration",
                           "screen_on_count", "screen_off_count"]

    values = [sensed_data.acc_x, sensed_data.acc_y, sensed_data.acc_z, sensed_data.acc_x_bef,
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
    return all_features_server, values, sensed_data.__dict__


def append_to_user_dataset(user_id, sensed_data, meal_taken):
    file_path = "ML/Saved Models/User Models/" + user_id + ".csv"
    headers, values, values_dict = feature_names_and_values(sensed_data=sensed_data)
    headers.append("meal_taken")
    headers.append("user_id")
    values.append(meal_taken)
    values_dict["meal_taken"] = meal_taken
    values_dict["user_id"] = user_id

    file = Path(file_path)
    if not file.exists():
        print("No File for user: ", user_id)
        with open(file_path, "w+", newline='') as file_data:
            writer = csv.DictWriter(file_data, delimiter=',', fieldnames=headers)
            writer.writeheader()

    print("Writing file: ", user_id)
    with open(file_path, "a", newline='') as file_data:
        writer = csv.DictWriter(file_data, delimiter=',', fieldnames=headers)
        writer.writerow(values_dict)

        # Append new data as a new raw

    # If number of rows > 10,
    #   train the model and save

    input_file = open(file_path, "r+")
    reader_file = csv.reader(input_file)
    file_len = len(list(reader_file)) - 1
    input_file.close()

    if file_len > 10:
        df = pd.read_csv(file_path)
        y_col = ["meal_taken"]
        f_groups = headers
        f_groups.remove("meal_taken")
        f_groups.remove("user_id")
        model_file_name = "User Models/" + user_id
        rf_results = BASE.train_and_save(dataframe=df, feature_group=f_groups, y_col=y_col, training_percentage_min=90,
                                         training_percentage_max=95, filename=model_file_name, pers= True)
        print(rf_results)
        return rf_results


def is_user_model_available(user_id):
    wbk_path = "ML/Saved Models/User Models/" + user_id + ".pkl"

    if os.path.isfile(wbk_path) and os.access(wbk_path, os.R_OK):
        print("[File] Model file available : " + wbk_path)
        return True
    else:
        print("[File] No such file : ", wbk_path)
        return False
