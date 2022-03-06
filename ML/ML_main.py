import ML.input as input
import ML.BASE as BASE

# import RandomForest as rf

df = input.df
x_cols = input.x_cols
y_col = input.y_col

# [FEATURE GROUPS] ==========================================================
f_groups = input.all_features_server

print("[Dataframe] Feature Groups      = ", f_groups)
# print("[Dataframe] Feature Group Names = ", f_groups_name_)

print("[Dataframe] # of features = ", len(f_groups))


# rf_results = BASE.train_and_save(dataframe=df, feature_group=f_groups, y_col=y_col, training_percentage_min=90,
#                                  training_percentage_max=95)
# print(rf_results)


# rf_predict = rf.rf_predict([[7, 7, 48.4, 0, 0, 0, 0, 0]], filename="rf_clf")
# print(rf_predict)

def train():
    rf_results = BASE.train_and_save(dataframe=df, feature_group=f_groups, y_col=y_col, training_percentage_min=90,
                                     training_percentage_max=95)
    print(rf_results)
    return rf_results
