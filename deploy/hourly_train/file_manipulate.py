import shutil
import os
import pandas as pd
from datetime import timedelta
import numpy as np

def move_file(old_path, new_path, pollutant, city):
    # filelist = os.listdir(old_path)
    filelist = []
    for i in range(10):
        file = "AutoML_{}_{}(gfs_{}).dat".format(pollutant, i+1, city)
        filelist.append(file)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)

def check_file(file_path):
    empty_file_list = []
    path = os.listdir(file_path)
    for p in path:
        if os.path.exists(file_path + "/" + p):
            if len(os.listdir(file_path + "/" + p)) == 0:
                empty_file_list.append(p)
    print(empty_file_list)
    print(len(path) - len(empty_file_list))


def remove_file(file_path):
    shutil.rmtree(file_path)


def pol_limit():
    """各污染物的上下限值"""
    pol_hour = {"SO2": [0, 150, 500, 650, 800, np.nan, np.nan, np.nan],
                "NO2": [0, 100, 200, 700, 1200, 2340, 3090, 3840],
                "CO": [0, 5, 10, 35, 60, 90, 120, 150],
                "O3": [0, 160, 200, 300, 400, 800, 1000, 1200],
                "PM2.5": [0, 35, 75, 115, 150, 250, 350, 500],
                "PM10": [0, 50, 150, 250, 350, 420, 500, 600]}
    pol_day = {"SO2": [0, 50, 150, 475, 800, 1600, 2100, 2620],
               "NO2": [0, 40, 80, 180, 280, 565, 750, 940],
               "CO": [0, 2, 4, 14, 24, 36, 48, 60],
               "O3": [0, 100, 160, 215, 265, 800, np.nan, np.nan],
               "PM2.5": [0, 35, 75, 115, 150, 250, 350, 500],
               "PM10": [0, 50, 150, 250, 350, 420, 500, 600]}
    return pol_hour, pol_day


def iaqi(y_pred, pollutant):
    """计算IAQI"""
    qua = [0, 50, 100, 150, 200, 300, 400, 500]  # 空气质量分指数

    pol_ = {}
    pol_hour, pol_day = pol_limit()
    if len(y_pred) > 10:
        pol_ = pol_hour
    else:
        pol_ = pol_day

    pred_iaqi_list = []
    # 预测值的IAQI
    print(len(y_pred[pollutant]))
    for value in y_pred[pollutant]:
        if (pollutant == "SO2") & (value > 800):
            pred_iaqi_list.append(np.nan)
            continue
        for i in range(len(qua) - 1):
            if (value >= pol_[pollutant][i]) & (value <= pol_[pollutant][i + 1]):
                iaqi = (qua[i + 1] - qua[i]) / (pol_[pollutant][i + 1] - pol_[pollutant][i]) * (
                        value - pol_[pollutant][i]) + qua[i]
                iaqi = round(iaqi, 2)
                pred_iaqi_list.append(iaqi)
    print(len(pred_iaqi_list))
    y_pred["{}_iaqi".format(pollutant)] = pred_iaqi_list


if __name__ == "__main__":
    # pollutant_list = ["SO2", "NO2", "CO", "O3", "PM2.5", "PM10"]
    # # pollutant_list = ["O3", "PM2.5", "PM10"]
    # city_list = ["chengdu", "leshan", "luzhou", "mianyang", "panzhihua", "yibin", "zigong"]
    # # for city in city_list:
    # #     for pollutant in pollutant_list:
    # move_file(r"/suncere/pyd/PycharmProjects/models/multi_station/{}".format("CO"),
    #           r"/suncere/pyd/PycharmProjects/models/sichuan_model/pred1(+GFS)/{}/{}".format("chengdu", "CO"),
    #           "CO", "chengdu")

    # os.mkdir("/suncere/pyd/PycharmProjects/北京")

    # file_path = "/usr/xgboost/GFS_data/{}06".format(pd.to_datetime(pd.datetime.now().strftime("%Y%m%d")) + timedelta(days=-1))
    # remove_file(file_path)

    # data = pd.read_csv("城市清单.csv")
    # print(data.head())
    # pd.set_option("display.max_rows", None)
    # for i in ["05", "06", "07", "08", "09"]:
    #     data = pd.read_csv("/mnt/output/AQMS/station/AQMS_hourly_202211{}.txt".format(i))
    #     data.columns = ["time", "code", "PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
    #     data = data.loc[(data["code"] == 510100066), :]
    #     data = data[:24]
    #     data.index = range(len(data))
    #     print(data)
    # pollutant_list = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
    # print([0]*72)
    # for pollutant in pollutant_list:
    #     iaqi(data, pollutant)
    # print(data)
    
    # for i in ["05","06", "07", "08", "09"]:
    #     pred_data = pd.read_csv("/mnt/real/AIR2/Station_2022-11-{}.tsv".format(i), sep="\t")
    #     # pred_data["TimePoint"] = pd.to_datetime(pred_data[["Year", "Month", "Day", "Hour"]])
    #     pred_data = pred_data.rename(columns={"O3-1h-24h": "O3", "Hour": "hour"})
    #     pred_data = pred_data.dropna(axis=0, how="any")
    #     pred_data["UniqueCode"] = pred_data["UniqueCode"].astype(int)
    #     pred_data = pred_data[
    #         ["hour", "UniqueCode", "SO2", "NO2", "CO", "O3", "PM2.5", "PM10"]]
    #     pred_data = pred_data[(pred_data["hour"].isin([23,0])) & (pred_data["UniqueCode"] == 510100066)]
    #     print(pred_data)

    station = pd.read_csv("station.csv")
    print(station.head())


