# feature Groups

# ACC
acc = ["acc_x", "acc_y", "acc_z"]
acc_bef = ["acc_x_bef", "acc_y_bef", "acc_z_bef"]
acc_aft = ["acc_x_aft", "acc_y_aft", "acc_z_aft"]
acc_abs = ["acc_xabs", "acc_yabs", "acc_zabs"]
acc_abs_bef = ["acc_xabs_bef", "acc_yabs_bef", "acc_zabs_bef"]
acc_abs_aft = ["acc_xabs_aft", "acc_yabs_aft", "acc_zabs_aft"]
acc_all = acc + acc_bef + acc_aft + acc_abs + acc_abs_bef + acc_abs_aft

# APP
app_facebook = ["app_facebook"]
app_whatsapp = ["app_whatsapp"]
app_googlequicksearchbox = ["app_googlequicksearchbox"]  # has null values
app_microsoft_launcher = ["app_microsoft_launcher"]
app_instagram = ["app_instagram"]
app_youtube = ["app_youtube"]
app_chrome = ["app_chrome"]
app_spotify = ["app_spotify"]  # has null values
app_android_dialer = ["app_android_dialer"]  # has null values
app_youtube_music = ["app_youtube_music"]
apps_usable = app_whatsapp + app_youtube + app_chrome + app_youtube_music + app_facebook + app_instagram + app_microsoft_launcher
apps_all = app_facebook + app_whatsapp + app_googlequicksearchbox + app_microsoft_launcher + app_instagram + app_youtube \
           + app_chrome + app_spotify + app_android_dialer + app_youtube_music

# BAT
bat = ['battery_level']  # has null values
charging_count = ["charging_true_count", "charging_false_count"]
charging_sources = ['charging_ac', 'charging_usb', 'charging_unknown']
bat_all = bat + charging_count + charging_sources
bat_usable = charging_count + charging_sources

# LOC
radius_of_gyration = ["radius_of_gyration"]
loc_all = radius_of_gyration

# TIME
time = ["month", "date", "minutes_elapsed"]
day = ["day"]
month = ["month"]
date = ["date"]
minutes_elapsed = ["minutes_elapsed"]
hours_elapsed = ["hours_elapsed"]
weekend = ['weekend']
time_all = minutes_elapsed + hours_elapsed + weekend

# SCR
screen_all = ['screen_on_count', 'screen_off_count']

# other
user_id = ["user_id"]

G1 = screen_all + apps_all + time_all # INT
G2 = acc_all + bat_all + time_all + loc_all # CONT
all_features = screen_all + apps_all + bat_all + time_all + acc_all + loc_all

all_features_without_null = screen_all + apps_usable + bat_usable + time_all + acc_all

# ###########################################################################3 feature selection

y_col = ["meal_taken"]
x_cols = screen_all + apps_all + bat_all + time_all + acc_all + loc_all
