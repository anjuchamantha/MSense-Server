import input
import BASE

df = input.df
x_cols = input.x_cols
y_col = input.y_col

# [FEATURE GROUPS] ==========================================================
f_groups = input.loc_all, input.screen_all, input.time_all, input.bat_all, input.apps_all, input.acc_all, input.G1, input.G2, input.all_features

f_groups_name = ["LOC", "SCR", "TIME", "BAT", "APP", "ACC",
                 "INTSEN", "CONSEN", "ALL"]

f_groups_ = input.screen_all + input.bat_all
f_groups_name_ = ["SCR", "BAT"]

print("[Dataframe] Feature Groups      = ", f_groups_)
print("[Dataframe] Feature Group Names = ", f_groups_name_)

print("[Dataframe] # of features = ", len(f_groups_))

rf_results = BASE.train_and_save(dataframe=df, feature_group=f_groups_, y_col=y_col)
print(rf_results)
