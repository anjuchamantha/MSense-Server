from datetime import datetime

import numpy as np
import pandas as pd

from ML import RandomForest


def split_user_base(df, training_percentage_min, training_percentage_max, pers):
    if not pers:
        # get the unique User Models in dataset
        usernames = df["user_id"].unique()

        # get 20% 25% bound for divide test data
        train_data_count_min = len(df.index) * training_percentage_min // 100
        train_data_count_max = len(df.index) * training_percentage_max // 100

        # create empty df
        train_data = pd.DataFrame(columns=df.columns)

        while True:
            # select random user from usernames arr and delete
            user_id = np.random.choice(usernames, 1, replace=False)[0]
            usernames = np.delete(usernames, np.argwhere(usernames == user_id))
            # get user dataframe
            user_dataframe = df[df['user_id'] == user_id]
            # add to user dataframes to train dataset
            if (len(train_data.index) + len(user_dataframe.index)) < train_data_count_min:  # len(train set) < 75%
                train_data = train_data.append(user_dataframe, ignore_index=True)

            elif train_data_count_min <= (len(train_data.index) + len(
                    user_dataframe.index)) <= train_data_count_max:  # 70% < len(train set) < 80%
                train_data = train_data.append(user_dataframe, ignore_index=True)
                break

            elif (len(train_data.index) + len(user_dataframe.index)) > train_data_count_max:  # 80% < len(train set)
                continue

        # y = float --> int
        train_data['meal_taken'] = train_data['meal_taken'].apply(np.int64)
        test_data = df[~df.isin(train_data)].dropna()

    else:
        train_data, test_data = RandomForest.test_train_split(df=df, train_size=0.9)
    return train_data, test_data


def train_and_save(dataframe, feature_group, y_col, filename, training_percentage_min=75,
                   training_percentage_max=80, pers=False):
    t1 = datetime.now()
    print(dataframe)
    # split test train split
    train_data, test_data = split_user_base(dataframe.copy(), training_percentage_min, training_percentage_max, pers)

    print(train_data)
    print(test_data)

    # ======= without smote
    x_train_ws = train_data[feature_group]
    y_train_ws = train_data[y_col]
    y_train_ws = y_train_ws.astype('int')
    # test data
    x_test_ws = test_data[feature_group]
    y_test_ws = test_data[y_col]
    y_test_ws = y_test_ws.astype('int')
    # results
    rf_score_ws = RandomForest.rf_train_and_save(x_train_ws, y_train_ws, x_test_ws, y_test_ws, filename=filename)

    t2 = datetime.now()
    # print("[Time] TRAIN Start Time    =", t1)
    print("[Time] TRAIN Time Elapsed    =", t2 - t1)

    return rf_score_ws
