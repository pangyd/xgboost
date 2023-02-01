# encoding:utf-8
import datetime
import logging
import pandas as pd
import numpy as np
from data_preprocess import *
from config import *
import pickle
import warnings
import matplotlib.pyplot as plt
import pymssql
import math
import time
from datetime import timedelta
import os
# from sqlalchemy import create_engine
import concat_gfs
import random


# 连接数据库
user = "sa"
password = "sun123!@"
host = "202.104.69.206:18710"
database1 = "EnvDataChina"
# database2 = "EnvDataChina_Release"
# conn = pymssql.connect(server=host, user=user, password=password, database=database2)
# cur = conn.cursor()
# engine = create_engine("mssql+pymssql://" + user + ":" + quote(password) + "@" + host + "/" + database)

warnings.filterwarnings("ignore")


def mean_abs_re(y_true, y_pred):
    return np.sum(abs((y_true - y_pred) / y_true)) / len(y_true)


def hourly_predict_data(pred_data, hourly_data, city, pollutant):
    # 加载模型
    model = pickle.load(
        open(
            "/usr/xgboost/models/{}/hourly_Xgboost_AutoML_{}.dat".format(
                city,
                pollutant),
            "rb"))

    for i in range(240):
        if i > 24:
            pred_data["{}_lastday".format(pollutant)][i+1] = pred_data["{}(1)".format(pollutant)][i-24+1]
            hourly_data["{}_lastday".format(pollutant)][i + 1] = hourly_data["{}(1)".format(pollutant)][i - 24 + 1]

        # 预测每一行的值，赋值给下一列
        x = pred_data[i:i+2]

        # 预测
        y_pred = model.predict(x)  # 单行不能预测
        pred_data["{}(1)".format(pollutant)][i + 2] = y_pred[-1]
        hourly_data["{}(1)".format(pollutant)][i + 2] = y_pred[-1]


def daily_pred_data(y_pred, daily_pred, pollutant):
    """污染物日均值"""
    for i in range(0, len(y_pred), 24):
        oneday_pred = list(y_pred[pollutant][i: i + 24])
        # 臭氧用八小时的滑动平均作为日均值
        if pollutant == "O3":
            sum_m, sum_n = 0, 0
            for j in range(len(oneday_pred) - 8):
                m = np.mean(oneday_pred[j: j + 8])
                sum_m += m
            daily_pred[pollutant][i / 24] = sum_m / 17
        else:
            daily_pred[pollutant][i / 24] = np.mean(oneday_pred)
    return daily_pred


def carry_ahead(number, pollutant):
    """四舍六入五成双"""
    for k in range(len(number)):
        if number[pollutant][k] != number[pollutant][k]:   # NAN自己不等于自己
            continue
        if number[pollutant][k] < (int(number[pollutant][k]) + 0.5):
            number[pollutant][k] = int(number[pollutant][k])
        if number[pollutant][k] > (int(number[pollutant][k]) + 0.5):
            number[pollutant][k] = int(number[pollutant][k]) + 1
        if number[pollutant][k] == (int(number[pollutant][k]) + 0.5):
            if (int(number[pollutant][k]) % 10) in [1, 3, 5, 7, 9]:
                number[pollutant][k] = int(number[pollutant][k]) + 1
            else:
                number[pollutant][k] = int(number[pollutant][k])
    return number[pollutant]


def add_quarter(data):
    """季节特征"""
    data["quarter"] = data["TimePoint"].dt.quarter
    for index in data.index:
        if data["month"][index] in [3, 4, 5]:
            data["quarter"][index] = 1
        if data["month"][index] in [6, 7, 8]:
            data["quarter"][index] = 2
        if data["quarter"][index] in [9, 10, 11]:
            data["quarter"][index] = 3
        if data["quarter"][index] in [1, 2, 12]:
            data["quarter"][index] = 4
    return data


def feature_select(pred_data, pollutant):
    """特征筛选"""
    if pollutant == "SO2":
        data = pred_data[["SO2_lastday", 'month', 'quarter', 'atmosphere_tcc', 'surface_SUNSD',
                               'surface_t', 'surface_prate', 'surface_gust', 'highCloudLayer_hcc',
                               'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_v',
                               'isobaric_700_u', 'isobaric_500_v', 'StationCode', 'SO2(1)']]
    if pollutant == "NO2":
        data = pred_data[['NO2_lastday', 'O3_lastday', 'month', 'quarter', 'atmosphere_tcc',
                               'surface_hpbl', 'surface_sp', 'surface_t', 'surface_prate',
                               'surface_gust', 'highCloudLayer_hcc', 'middleCloudLayer_mcc',
                               'lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_700_v',
                               'isobaric_500_gh', 'isobaric_500_v', 'StationCode', 'NO2(1)']]
    if pollutant == "CO":
        data = pred_data[['CO_lastday', 'O3_lastday', 'month', 'quarter', 'hour',
                               'atmosphere_tcc', 'surface_hpbl', 'surface_SUNSD', 'surface_t',
                               'surface_gust', 'highCloudLayer_hcc', 'middleCloudLayer_mcc',
                               'lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_700_u',
                               'isobaric_500_gh', 'isobaric_500_u', 'StationCode', 'CO(1)']]
    if pollutant == "O3":
        data = pred_data[['O3_lastyear', 'O3_lastday', 'month', 'quarter', 'hour',
                               'surface_hpbl', 'surface_sp', 'surface_SUNSD', 'surface_t',
                               'surface_gust', 'highCloudLayer_hcc', 'middleCloudLayer_mcc',
                               'lowCloudLayer_lcc', 'isobaric_500_gh', 'StationCode', 'O3(1)']]
    if pollutant == "PM2.5":
        data = pred_data[['PM2.5_lastday', 'month', 'quarter', 'atmosphere_tcc', 'surface_sp',
                               'surface_t', 'surface_prate', 'surface_gust', 'highCloudLayer_hcc',
                               'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_v',
                               'isobaric_700_u', 'isobaric_500_gh', 'StationCode', 'PM2.5(1)']]
    if pollutant == "PM10":
        data = pred_data[['PM10_lastday', 'month', 'quarter', 'atmosphere_tcc', 'surface_sp',
                               'surface_t', 'surface_prate', 'surface_gust', 'highCloudLayer_hcc',
                               'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_v',
                               'isobaric_700_u', 'isobaric_500_gh', 'StationCode', 'PM10(1)']]

    return data


def get_lastyear_data():
    now_data = pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d"))
    d_list = []
    for i in range(11):
        try:
            d = pd.read_csv("/mnt/real/AIR/Station_{}-{}-{}.tsv".format(
                now_data.year - 1, now_data.month, str(now_data.day + i -1).rjust(2, "0")), sep="\t")
        except:
            d = pd.read_csv("/mnt/real/AIR/Station_{}-{}-{}.csv".format(
                now_data.year - 1, now_data.month, str(now_data.day + i -1).rjust(2, "0")), sep="\t")
        finally:
            d_list.append(d)
    return d_list


def get_data():
    # 获取时间列：前一晚23:00~十天后00:00
    now_data = pd.datetime.now().strftime("%Y-%m-%d %H")
    now_data = pd.to_datetime(now_data)
    timepoint = [now_data + timedelta(hours=-2) + timedelta(hours=+i) for i in range(242)]
    timepoint = pd.DataFrame(timepoint, columns=["TimePoint"])
    # timepoint["TimePoint"] = timepoint["TimePoint"].apply(lambda x: x.strftime("%Y%m%d%H"))
    timepoint["hour"] = timepoint["TimePoint"].dt.hour
    timepoint["month"] = timepoint["TimePoint"].dt.month
    timepoint = add_quarter(timepoint)

    # 获取前一天的污染物数据
    pred_data = pd.read_csv(
        "/mnt/real/AIR2/Station_{}.tsv".format(
            (pd.to_datetime(pd.datetime.now()) + timedelta(days=-1)).strftime("%Y-%m-%d")), sep="\t")
    pred_data["TimePoint"] = pd.to_datetime(pred_data[["Year", "Month", "Day", "Hour"]])
    pred_data = pred_data.rename(columns={"O3-1h-24h": "O3", "Hour": "hour"})
    pred_data = pred_data.dropna(axis=0, how="any")
    pred_data["UniqueCode"] = pred_data["UniqueCode"].astype(int)
    pred_data = pred_data[
        ["TimePoint", "Area", "StationCode", "UniqueCode", "SO2", "NO2", "CO", "O3", "PM2.5", "PM10", "hour"]]

    # 获取上上天的数据
    last_day = pd.read_csv(
        "/mnt/real/AIR2/Station_{}.tsv".format(
            (pd.to_datetime(pd.datetime.now()) + timedelta(days=-2)).strftime("%Y-%m-%d")), sep="\t")
    last_day = last_day.rename(
        columns={"SO2": "SO2_lastday", "NO2": "NO2_lastday", "CO": "CO_lastday", "O3-1h-24h": "O3_lastday",
                 "PM2.5": "PM2.5_lastday", "PM10": "PM10_lastday", "Hour": "hour"})
    last_day = last_day.dropna(axis=0, how="any")
    last_day["UniqueCode"] = last_day["UniqueCode"].astype(int)
    # 加入前一天同时间点的数据
    last_day = last_day[["UniqueCode", "SO2_lastday", "NO2_lastday", "CO_lastday",
                         "O3_lastday", "PM2.5_lastday", "PM10_lastday", "hour"]]

    # 合并两天的数据
    pred_data = pd.merge(pred_data, last_day, on=["UniqueCode", "hour"], how="outer")
    sichuan_city = ["成都市", "乐山市", "泸州市", "绵阳市", "攀枝花市", "宜宾市", "自贡市"]
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "成都市"].index, axis=0)
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "乐山市"].index, axis=0)
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "泸州市"].index, axis=0)
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "绵阳市"].index, axis=0)
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "攀枝花市"].index, axis=0)
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "宜宾市"].index, axis=0)
    pred_data = pred_data.drop(labels=pred_data[pred_data["Area"] == "自贡市"].index, axis=0)

    # 按照时间和站点排序
    pred_data.sort_values(by=["UniqueCode", "TimePoint"], inplace=True)

    # 只取晚上23点和00点的数据作为预测数据
    data = pred_data[(pred_data["hour"] == 23) | (pred_data["hour"] == 0)]
    data.index = range(len(data))

    # 按照站点编号分组，分别处理每一个站点
    data = data.groupby(by="StationCode")
    pred_data = pred_data.groupby(by="StationCode")

    return timepoint, data, pred_data

if __name__ == "__main__":
    start_time = time.time()
    # 创建日志
    filename = "all_city_output"
    logging.basicConfig(filename="/usr/xgboost/deploy/" + filename + ".log",
                        filemode="a", level=logging.INFO,
                        format="%(message)s")

    # O3上一年同时段的数据
    d_list = get_lastyear_data()
    # 按照时间进行排序
    for d in d_list:
        d.rename(columns={"O3-1h-24h": "O3"}, inplace=True)
        d = d[["Hour", "StationCode", "O3"]]

    pollutant_list = ["SO2", "NO2", "CO", "O3", "PM2.5", "PM10"]

    timepoint, data, la = get_data()   # 监测数据
    gfs_data = concat_gfs.all_timepoint   # GFS(已经分组)
    d1 = concat_gfs.d1
    d2 = concat_gfs.d2
    # 查看当前运行了多少个站点
    k = 0
    for (code, group), (code1, group1) in zip(data, la):
        group.index = range(len(group))

        t1 = time.time()

        # 判断该站点所在的城市是否存在模型,存在则预测，不存在跳过该站点
        if os.path.exists("/usr/xgboost/models/{}".format(group["Area"][0])):
            if len(os.listdir("/usr/xgboost/models/{}".format(group["Area"][0]))) != 0:
                group = group.drop(labels=["hour"], axis=1)
                hourly_data = timepoint.merge(group, on="TimePoint", how="outer")   # 还差GFS
                # GFS数据加上时间
                try:
                    single_station_gfs = gfs_data.get_group(code)
                    single_station_gfs.index = range(len(single_station_gfs))
                    sole_station = pd.concat([d1, single_station_gfs[:81]], axis=1)
                    sole_station = d2.merge(sole_station, how="outer", on="TimePoint")
                    sole_station = sole_station.interpolate(method="ffill")
                    sole_station = sole_station.drop(labels=["Area", "StationCode", "UniqueCode"], axis=1)
                except:
                    continue
                else:
                    # 合并监测数据和GFS数据
                    hourly_data = hourly_data.merge(sole_station, on="TimePoint", how="outer")

                    # 线性插值
                    hourly_data[0: 2] = hourly_data[0: 2].interpolate(method="ffill", limit_direction="forward")
                    hourly_data[0: 2] = hourly_data[0: 2].interpolate(method="bfill", limit_direction="backward")
                    # 补齐站点编号
                    hourly_data[["Area", "StationCode", "UniqueCode"]] = \
                        hourly_data[["Area", "StationCode", "UniqueCode"]].interpolate(method="ffill")
                    # 补齐前一天的数据
                    for p in pollutant_list:
                        lastday = list(group1[p])
                        group1[p] = group1[p].interpolate(method="linear", limit_direction="forward")
                        group1[p] = group1[p].interpolate(method="linear", limit_direction="backward")
                        if len(lastday) < 24:
                            for i in range(24 - len(lastday)):
                                lastday.append(lastday[-1])
                        if len(lastday) == 24:
                            hourly_data["{}_lastday".format(p)][2:2+24] = lastday
                        # 污染物数据前两行都为空时
                        if hourly_data[p][0] == np.nan:
                            hourly_data[p][0] = hourly_data["{}_lastday".format(p)][0]
                        if hourly_data[p][1] == np.nan:
                            hourly_data[p][1] = hourly_data["{}_lastday".format(p)][1]

                    # O3加入上一年同时间的数据
                    hourly_data["O3_lastyear"] = 0   # 初始化为0
                    d = d_list[0][d_list[0]["StationCode"] == code]
                    if (23 in d["Hour"]) & (0 in d["Hour"]):
                        hourly_data["O3_lastyear"][:2] = d["O3"][d["Hour"].isin([23, 0])]
                    else:
                        hourly_data["O3_lastyear"][:2] = hourly_data["O3_lastday"][:2]
                    for i, d in zip(np.arange(2, 219, 24), d_list[1:]):
                        d["O3"] = d["O3"].interpolate(method="linear", limit_direction="forward")
                        d["O3"] = d["O3"].interpolate(method="linear", limit_direction="backward")
                        lastyear = list(d["O3"][d["StationCode"] == code])
                        if len(lastyear) == 0:
                            lastyear = random.sample(range(10, 110), 24)
                        if 0 < len(lastyear) < 24:
                            for i in range(24 - len(lastyear)):
                                lastyear.append(lastyear[-1])
                        if len(lastyear) == 24:
                            hourly_data["O3_lastyear"][i: i + 24] = lastyear
                    hourly_data["O3_lastyear"] = hourly_data["O3_lastyear"].interpolate(method="linear", limit_direction="forward")
                    hourly_data["O3_lastyear"] = hourly_data["O3_lastyear"].interpolate(method="linear", limit_direction="backward")

                    for column in hourly_data.columns:
                        hourly_data[column][hourly_data[column] == "—"] = np.nan

                    # 把站点列转换为整型
                    hourly_data["StationCode"] = hourly_data["StationCode"].str.replace("\D+", "").astype(int)

                    # 把污染物列变为浮点型
                    hourly_data[["SO2", "NO2", "CO", "O3", "PM2.5", "PM10", "SO2_lastday", "NO2_lastday",
                                 "CO_lastday", "O3_lastday", "PM2.5_lastday", "PM10_lastday"]] = \
                    hourly_data[["SO2", "NO2", "CO", "O3", "PM2.5", "PM10", "SO2_lastday", "NO2_lastday",
                                 "CO_lastday", "O3_lastday", "PM2.5_lastday", "PM10_lastday"]].astype(float)

                    # 前一天数据线性插值
                    hourly_data[["SO2_lastday", "NO2_lastday", "CO_lastday", "O3_lastday", "PM2.5_lastday", "PM10_lastday"]][
                    :26] = \
                        hourly_data[
                            ["SO2_lastday", "NO2_lastday", "CO_lastday", "O3_lastday", "PM2.5_lastday", "PM10_lastday"]][
                        :26].interpolate(method="linear", limit_direction="forward")

                    hourly_data.rename(columns={"SO2": "SO2(1)", "NO2": "NO2(1)", "CO": "CO(1)", "O3": "O3(1)", "PM2.5": "PM2.5(1)", "PM10": "PM10(1)"}, inplace=True)

                    # if (hourly_data["TimePoint"][241].month == 1) & (hourly_data["TimePoint"][241].day in np.arange(1, 11)):
                    #     hourly_data["O3_lastyear"] = hourly_data["O3_lastday"]
                    # else:
                    #     try:
                    #         从数据库调取上一年数据，速度慢
                    #         O3_lastyear = pd.read_sql("Air_h_{}_{}_App".format(int(pd.datetime.now().strftime("%Y"))-1, code), engine)
                    #         O3_lastyear["TimePoint"] = O3_lastyear["TimePoint"].apply(lambda x: x.strftime("%Y-%m-%d %H:%m:%s"))
                    #         date1 = hourly_data["TimePoint"] >= str(hourly_data["TimePoint"][0])
                    #         date2 = hourly_data["TimePoint"] <= str(hourly_data["TimePoint"][-1])
                    #         hourly_data["O3_lastyear"] = O3_lastyear["O3"][O3_lastyear["TimePoint"].isin(list(O3_lastyear[date1 & date2]["TimePoint"]))]
                    #         hourly_data["O3_lastyear"] = hourly_data["O3_lastyear"].interpolate(method="linear", limit_direction="forward")
                    #         hourly_data["O3_lastyear"] = hourly_data["O3_lastyear"].interpolate(method="linear", limit_direction="backward")
                    #     except:
                    #         hourly_data["O3_lastyear"] = hourly_data["O3_lastday"]

                    # 单个污染物分别预测
                    for pollutant in pollutant_list:
                        pollutant_data = feature_select(hourly_data, pollutant)
                        hourly_predict_data(pollutant_data, hourly_data, "{}".format(group["Area"][0]), pollutant)   # 预测
                        hourly_data["{}(1)".format(pollutant)] = carry_ahead(hourly_data, "{}(1)".format(pollutant))   # 四舍六入五成双
                        hourly_data.rename(columns={"{}(1)".format(pollutant): "{}".format(pollutant)}, inplace=True)
                        logging.info(f"{pollutant}小时值预测结果:{list(hourly_data[pollutant])}")

                    hourly_predict_pol = hourly_data[["TimePoint", "UniqueCode", "SO2", "NO2", "CO", "O3", "PM2.5", "PM10"]][2:]
                    hourly_predict_pol["TimePoint"] = hourly_predict_pol["TimePoint"].apply(lambda x: x.strftime("%Y%m%d%H"))
                    hourly_predict_pol["UniqueCode"] = hourly_predict_pol["UniqueCode"].astype(int)
                    hourly_predict_pol.index = range(len(hourly_predict_pol))
                    k += 1
                    print("第{}个站点".format(i))
                    try:
                        hourly_predict_pol.to_csv(
                            "/mnt/output/suncere/station/Xgboost_hourly_{}.tsv".format(pd.datetime.now().strftime("%Y%m%d")),
                            sep=",", mode="a", header=0, index=0)
                        hourly_predict_pol.to_csv(
                            "/mnt/output/suncere/city/Xgboost_hourly_{}.tsv".format(pd.datetime.now().strftime("%Y%m%d")),
                            sep=",", mode="a", header=0, index=0)
                    except:
                        pass

                    # 污染物日均值
                    daily_pred = pd.DataFrame(columns=hourly_predict_pol.columns, index=range(10))
                    daily_pred["TimePoint"] = [pd.to_datetime(pd.datetime.now().strftime("%Y%m%d")) + timedelta(days=+i) for i in range(10)]
                    daily_pred["UniqueCode"] = [group["UniqueCode"][0]] * 10

                    for pollutant in hourly_predict_pol.columns[2:]:

                        # 计算浓度日均值
                        daily_pred = daily_pred_data(hourly_predict_pol, daily_pred, pollutant)

                        # 四舍六入五成双
                        # 日均值
                        daily_pred[pollutant] = carry_ahead(daily_pred, pollutant)
                        logging.info(f"{pollutant}日均值预测结果:{list(daily_pred[pollutant])}")
                    try:
                        daily_pred.to_csv(
                            "/mnt/output/suncere/station/Xgboost_daily_{}.tsv".format(pd.datetime.now().strftime("%Y%m%d")),
                            sep=",", mode="a", header=0, index=0)
                        daily_pred.to_csv(
                            "/mnt/output/suncere/city/Xgboost_daily_{}.tsv".format(pd.datetime.now().strftime("%Y%m%d")),
                            sep=",", mode="a", header=0, index=0)
                    except:
                        pass

        else:
            continue
        t2 = time.time()
        print("单个城市的预测时间：{}s".format(t2-t1))

    end_time = time.time()
    logging.info(f"所花费的时间：{end_time - start_time}s")
    print("花费时间：{}h".format((end_time - start_time)/3600))



