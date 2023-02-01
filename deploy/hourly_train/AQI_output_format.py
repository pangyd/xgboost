import pandas as pd
from config import *

# 连接数据库
# user = "sa"
# password = "sun123!@"
# host = "202.104.69.206:18710"
# database = "EnvDataChina"
# engine = create_engine("mssql+pymssql://" + user + ":" + quote(password) + "@" + host + "/" + database)
#
# station_information = pd.read_sql("Station", engine)
#
#
# # AQI输出的形式
# pred_data = pd.DataFrame(columns=["Time", "Station", "PM2.5", "PM10", "CO", "NO2", "SO2", "O3"], index=range(24*10))
# timepoint = pd.datetime.now().strftime("%Y%m%d")
# pred_data["Station"] = "110000244"
# t = str(timepoint) + "00"
# for i in range(24*10):
#     pred_data["Time"][i] = str(int(t) + i)
#     pred_data["SO2"][i] = i
#     pred_data["NO2"][i] = i
#     pred_data["CO"][i] = i
#     pred_data["O3"][i] = i
#     pred_data["PM2.5"][i] = i
#     pred_data["PM10"][i] = i
# pred_data.to_csv("modelname_hourly_{}".format(pd.datetime.now().strftime("%Y%m%d")), sep=",", index=False)

data = pd.read_excel("citys.xlsx")
print(data.head())

