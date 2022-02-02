from datetime import datetime

import numpy as np
import pandas as pd

from classifiers import RandomForest
from utils import tools


def method_1_split_user_specific(df, user_id):
    # get user data and remove from the dataset
    user_df = df[df['user_id'] == user_id]
    df = tools.df_without_selected_user(df=df, user=user_id)

    # get the unique users in dataset
    usernames = df["user_id"].unique()

    # get 20% 25% bound for divide test data
    train_data_count_min = len(df.index) * 75 // 100
    train_data_count_max = len(df.index) * 80 // 100

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

    return train_data, user_df


def method_1_user_specific_iteration(dataframe, user_id, feature_group, feature_group_name, y_col, iteration_count=10):
    rf_list_ws, rf_list_s = [], []

    t1 = datetime.now()
    print("\n[FEATURE GROUP] " + feature_group_name)
    print("[Time] Feature Group Start Time    =", t1)

    for i in range(iteration_count):
        print("\n\nIteration = ", i + 1)

        # split test train split
        train_data, test_data = method_1_split_user_specific(dataframe.copy(), user_id)

        # ======= without smote
        x_train_ws = train_data[feature_group]
        y_train_ws = train_data[y_col]
        y_train_ws = y_train_ws.astype('int')
        # test data
        x_test_ws = test_data[feature_group]
        y_test_ws = test_data[y_col]
        y_test_ws = y_test_ws.astype('int')
        # results
        rf_score_ws = RandomForest.rf(x_train_ws, y_train_ws, x_test_ws, y_test_ws)
        rf_list_ws.append(rf_score_ws)
        # ===================
        # ======== with smote
        x_train_s, y_train_s = tools.random_over_under_sampling_2_class(df=train_data)
        x_train_s = x_train_s[feature_group]
        y_train_s = y_train_s[y_col]
        y_train_s = y_train_s.astype('int')
        # test data
        x_test_s, y_test_s = tools.under_sampling_2_class(df=test_data)
        x_test_s = x_test_s[feature_group]
        y_test_s = y_test_s[y_col]
        # results
        rf_score_s = RandomForest.rf(x_train_s, y_train_s, x_test_s, y_test_s)
        rf_list_s.append(rf_score_s)

    t2 = datetime.now()
    print("\n[FEATURE GROUP] " + feature_group_name)
    print("[Time] Feature Group Start Time    =", t1)
    print("[Time] Feature Group End Time    =", t2)
    print("[Time] Time Elapsed    =", t2 - t1)

    rf_data_ws = tools.print_mean_sd_median_min_max_of_confusion_matrix_values("RF ", rf_list_ws)
    rf_data_s = tools.print_mean_sd_median_min_max_of_confusion_matrix_values("RF ", rf_list_s)
    return rf_data_ws + rf_data_s


def base(dataframe, y_col, f_groups, f_group_names, filename, number_of_users=10):
    print("\n\n[BASE] Running Method 1 with top %s users" % number_of_users.__str__())
    dataframe = tools.normalize(dataframe)

    user_counts_df = tools.top_users(dataframe).rename_axis('user_id').reset_index(name='counts')
    print(user_counts_df.head(number_of_users))
    selected_users = user_counts_df[0:61]

    for index, row in selected_users.iterrows():
        user = row['user_id']
        user_df = dataframe[dataframe['user_id'] == user]

        eat_count = len(user_df[user_df[y_col[0]] > 0])
        non_eat_count = len(user_df[user_df[y_col[0]] == 0])
        eat_percentage = eat_count * 100 / float(len(user_df.index))
        non_eat_percentage = non_eat_count * 100 / float(len(user_df.index))

        print("\n\n[USER] Current User : " + str(user) + " : " + str(row["counts"]) + " rows")
        print("Eating percentage     :", eat_percentage)
        print("non-eating percentage :", non_eat_percentage)
        t1 = datetime.now()
        print("\n[Time] User Start Time    =", t1)

        # single excel sheet for single user
        excel_sheet = []

        excel_first_row = ["F Groups",
                           "#rows", "#eat", "eat%", "#n-eat", "n-eat%",

                           "RF_acc", "sd", "median", "min", "max",
                           "RF_f1_w", "sd", "median", "min", "max",
                           "RF_f1_a", "sd", "median", "min", "max",
                           "RF_auc", "sd", "median", "min", "max",
                           "RF_kappa", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",

                           "NB_accuracy", "sd", "median", "min", "max",
                           "NB_f1_w", "sd", "median", "min", "max",
                           "NB_f1", "sd", "median", "min", "max",
                           "NB_auc", "sd", "median", "min", "max",
                           "NB_kappa", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",

                           "GB_accuracy", "sd", "median", "min", "max",
                           "GB_f1_w", "sd", "median", "min", "max",
                           "GB_f1", "sd", "median", "min", "max",
                           "GB_auc", "sd", "median", "min", "max",
                           "GB_kappa", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",

                           "MLP_a", "sd", "median", "min", "max",
                           "MLP_f_w", "sd", "median", "min", "max",
                           "MLP_f", "sd", "median", "min", "max",
                           "MLP_a", "sd", "median", "min", "max",
                           "MLP_k", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",

                           "SVM_accuracy", "sd", "median", "min", "max",
                           "SVM_f1_w", "sd", "median", "min", "max",
                           "SVM_f1", "sd", "median", "min", "max",
                           "SVM_auc", "sd", "median", "min", "max",
                           "SVM_kappa", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",

                           "XGB_accuracy", "sd", "median", "min", "max",
                           "XGB_f1_w", "sd", "median", "min", "max",
                           "XGB_f1", "sd", "median", "min", "max",
                           "XGB_auc", "sd", "median", "min", "max",
                           "XGB_kappa", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",

                           "ADAB_accuracy", "sd", "median", "min", "max",
                           "ADAB_f1_w", "sd", "median", "min", "max",
                           "ADAB_f1", "sd", "median", "min", "max",
                           "ADAB_auc", "sd", "median", "min", "max",
                           "ADAB_kappa", "sd", "median", "min", "max",
                           "tp", "fn", "fp", "tn", "---",
                           ]

        # excel_first_row = ["F Groups",
        #                    '#rows', "#eat", "eat%", "#n-eat", "n-eat%",
        #
        #                    "WS-acc", "sd", "median", "min", "max",
        #                    "WS-f1_w", "sd", "median", "min", "max",
        #                    "WS-f1_avg", "sd", "median", "min", "max",
        #                    "WS-auc_1", "sd", "median", "min", "max",
        #                    "WS-auc_0", "sd", "median", "min", "max",
        #                    "WS-kappa", "sd", "median", "min", "max",
        #                    "tp", "fn", "fp", "tn",
        #
        #                    "acc", "sd", "median", "min", "max",
        #                    "f1_w", "sd", "median", "min", "max",
        #                    "f1_avg", "sd", "median", "min", "max",
        #                    "auc-1", "sd", "median", "min", "max",
        #                    "auc-0", "sd", "median", "min", "max",
        #                    "kappa", "sd", "median", "min", "max",
        #                    "tp", "fn", "fp", "tn"
        #                    ]

        excel_sheet.append(excel_first_row)

        for i in range(len(f_groups)):
            print("\n\n[USER] :", index + 1, ":", user)
            print("[FEATURE GROUP] : " + f_group_names[i])
            user_result_row = [f_group_names[i], row["counts"], eat_count, eat_percentage, non_eat_count,
                               non_eat_percentage] + method_1_user_specific_iteration(dataframe=dataframe,
                                                                                      user_id=user,
                                                                                      feature_group=f_groups[i],
                                                                                      feature_group_name=f_group_names[
                                                                                          i],
                                                                                      y_col=y_col)

            print(user_result_row)
            excel_sheet.append(user_result_row)

        tools.write_to_excel(excel_sheet, filename=filename, sheet_name=str(index + 1) + user[:5])

        t2 = datetime.now()
        print("\n\n[USER] Current User : " + str(user) + " : " + str(row["counts"]) + " rows")
        print("[Time] User Start Time    =", t1)
        print("[Time] User End Time    =", t2)
        print("[Time] Time elapsed    =", t2 - t1)
