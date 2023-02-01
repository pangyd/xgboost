import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote
import warnings
import os
from config import *
import shutil

warnings.filterwarnings("ignore")

# 连接数据库
user = "sa"
password = "sun123!@"
host = "202.104.69.206:18710"
database = "EnvDataChina"

engine = create_engine("mssql+pymssql://" + user + ":" + quote(password) + "@" + host + "/" + database)
station = pd.read_sql("Station", engine)
# pd.set_option("display.width", None)

pol, window1, window2, model_parameter = connect_database()

# 删除非国控站点
station = station[station["StationTypeId"] == 1]
# 删除未知城市,并按照站点排序
station["station_num"] = None
for i in station.index:
    station["station_num"][i] = int(station["StationCode"][i].strip("A"))
    if "市" not in station["Area"][i]:
        station = station.drop(i, axis=0)
station = station.sort_values(by="station_num", axis=0)
station.index = range(len(station))

# 站点按照城市进行分组
grouped = station.groupby(["Area"], sort=False)

station_dict = {}
for label, group in grouped:
    station_dict[label] = list(group["StationCode"].values)

# 删除四川的城市
sichuan_city = ["成都市", "乐山市", "泸州市", "绵阳市", "攀枝花市", "宜宾市", "自贡市"]
for city in sichuan_city:
    station_dict.pop(city)
print(station_dict)

# 创建文件夹
# for key in station_dict:
#     path1 = "/suncere/pyd/PycharmProjects/models/{}".format(key)
#     if os.path.exists(path1):
#         shutil.rmtree(path1)
#     os.mkdir(path1)
#     path2 = "/suncere/pyd/PycharmProjects/output/all_city_train/{}".format(key)
#     if os.path.exists(path2):
#         shutil.rmtree(path2)
#     os.mkdir(path2)


