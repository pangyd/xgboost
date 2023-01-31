import numpy as np
import pandas as pd
from data_preprocess import *
from config import *
from all_city_station import *
import PM10_flaml_automl
import logging
import time

warnings.filterwarnings("ignore")

"""处理每个城市的所有国控站点数据（除四川外，其他城市全部按照成都的方案进行训练），存在缺失特征则忽略该特征，同一个城市用同一套模型"""

# 连接数据库
user = "sa"
password = "sun123!@"
host = "202.104.69.206:18710"
database = "EnvDataChina"

engine1 = create_engine("mssql+pymssql://" + user + ":" + quote(password) + "@" + host + "/" + database)

# GFS的时间字符串（个位数前面保留0）
date = pd.date_range(start="2021-01-01", end="2022-09-01", freq="D")
mon = date.month.map(lambda x: "0"+str(x) if len(str(x)) == 1 else str(x))
da = date.day.map(lambda x: "0"+str(x) if len(str(x)) == 1 else str(x))
date_list = []
for i in range(len(date)):
    y = str(date.year[i])
    m = mon[i]
    d = da[i]
    date_list.append(y+m+d)
date = date_list

hour = ["000", "003", "006", "009", "012", "015", "018", "021"]
datetime = pd.date_range(start="2021-01-01 08:00:00", end="2022-09-01 05:00:00", freq="3H")
datetime = pd.DataFrame(datetime, columns=["TimePoint"])
datetime1 = pd.date_range(start="2021-01-01 08:00:00", end="2022-09-01 05:00:00", freq="H")
datetime1 = pd.DataFrame(datetime1, columns=["TimePoint"])

# 参数字典
pol, window1, window2, model_parameter = connect_database()

years = [2021, 2022]

# 污染物编号
code = ["100", "101", "102", "103", "104", "105"]
# feature_dict = {"100": "SO2", "101": "NO2", "102": "O3", "103": "CO", "104": "PM10", "105": "PM2.5",
#                 "108": "wind_spd", "109": "wind_dir", "110": "press", "111": "tem", "112": "hum"}
feature_dict = {"100": "SO2", "101": "NO2", "102": "O3", "103": "CO", "104": "PM10", "105": "PM2.5"}

# 监测数据特征列表
feature_list = []
for key in feature_dict:
    feature_list.append(feature_dict[key])

for city in station_dict:
    single_city_data = pd.DataFrame()
    # 查看当前在训练哪个城市
    print(city)
    # 提取站点编号的数字字段
    station_num = []
    for string in station_dict[city]:
        a = ""
        for i in string:
            if str.isdigit(i):
                a += i
        station_num.append(int(a))

    # 一个城市多个站点共用一套模型
    for number, station in zip(station_num, station_dict[city]):
        data = pd.DataFrame()
        for year in years:
            if year == 2022:
                year_data = pd.date_range(start="2022-01-01 08:00:00", end="2022-09-01 05:00:00", freq="H")
                year_data = pd.DataFrame(year_data, columns=["TimePoint"])
            else:
                year_data = pd.date_range(start="{}-01-01 08:00:00".format(year),
                                          end="{}-01-01 08:00:00".format(year + 1),
                                          freq="H")
                year_data = pd.DataFrame(year_data, columns=["TimePoint"])

            # 如果表不存在则跳过
            try:
                df = pd.read_sql("Air_h_{}_{}_App".format(year, station), engine1)  # 单个站点
            except Exception as result:
                break
            else:
                # 判断表是否为空表或缺少污染物，是则退出内循环
                if df.empty | (((set(code) < set(df["PollutantCode"])) | (set(code) == set(df["PollutantCode"]))) is False):
                    break

                # 把异常值标识为nan
                if "H" in df["Mark"]:
                    df["Mark"][df["Mark"] == "H"] = ""
                df["MonValue"][df["Mark"] != ""] = np.nan

                # 按照污染物分组，按照时间合并
                grouped = df.groupby(by="PollutantCode")
                label_list = []
                for label, group in grouped:  # label:[str]
                    label_list.append(label)
                    if label in feature_dict.keys():
                        group = group[["TimePoint", "MonValue"]]
                        # group = group.set_index("TimePoint")
                        group = group.rename(columns={"MonValue": "{}".format(feature_dict[label])})
                        year_data = year_data.merge(group, on="TimePoint", how="outer")

                        # [SO2, NO2, O3, PM2.5, PM10]都×1000
                        if feature_dict[label] in ["SO2", "NO2", "O3", "PM2.5", "PM10"]:
                            year_data[feature_dict[label]] = year_data[feature_dict[label]] * 1000
                    # 若label不在污染物列表内，则删除该组
                    else:
                        useless_label = grouped.get_group(label).index
                        df = df.drop(useless_label)

                # 转换列名，拼接每年数据
                # data.columns = year_data.columns
                data = data.append(year_data)

        # 如果是空表则跳过该站点
        try:
            pd.read_sql("Air_h_2021_{}_App".format(station), engine1)
            pd.read_sql("Air_h_2022_{}_App".format(station), engine1)
        except Exception as result:
            pass
        else:
            # 如果是空表则退出此轮循环，并在列表中删除此站点
            if df.empty | (((set(code) < set(df["PollutantCode"])) | (set(code) == set(df["PollutantCode"]))) is False):
                station_dict[city].remove(station)
                station_num.remove(number)
                continue

            # 查看当前处理的站点
            print(station)

            # 保留需要的时间段的数据
            data["TimePoint"] = pd.to_datetime(data["TimePoint"])
            date1 = data["TimePoint"] >= "2021-01-01 08:00:00"
            date2 = data["TimePoint"] <= "2022-09-01 05:00:00"
            data = data[data["TimePoint"].isin(list(data[date1 & date2]["TimePoint"]))]
            data.index = range(len(data))
            # object -> float64
            data[data.columns[1:]] = data[data.columns[1:]].astype("float64")

            # 把六大污染物<0的值转换成nan
            for column in data.columns[1:7]:
                data[column][data[column] <= 0] = np.nan
            # print(data.isnull().sum(axis=0))   # [34, 43, 33, 51, 68, 62, 200, 540, 2629, 147, 154, 1949, 147]

            # 处理缺失值
            # data.iloc[:, 1:9] = data.iloc[:, 1:9].interpolate(method="linear")
            # data.iloc[:, 9:] = data.iloc[:, 9:].interpolate(method="linear")
            # data.dropna(axis=0, how="any",inplace=True)

            """特征工程"""
            # 平滑滤波
            # data = filter(args, data)   # 只对一个污染物平滑滤波

            # 异常值处理
            # data = isolation_forest(data)
            # data = lof_detection(data)
            data = var_detection(data)

            # 加入相同时间点的数据
            # O3
            # data = add_lastyear_data(data, pol['pollutant4'])
            data = add_lastday_data(data, pol['pollutant6'])
            # data = add_lastday_data(data, pol['pollutant4'])
            # data = add_lastweek_data(data)

            # 加入当天的最高和最低温
            # data = max_min_tem(data)

            # 加入当天最高气压和最低气压
            # data = max_min_press(data)

            # 加入6h和12h的极大风速
            # data = max_spd_6h_12h(data)
            # 前一天变压
            # data = press_change(data)

            # 风向风速处理
            # data = wind_spd_dir(data)
            # data = onehot(data)
            # data = emb_layer(data)

            # 季节、月份特征提取
            data = time_feature(data)
            data = hour_data(data)

            # 主成分分析
            # data = PCA(n_components=6)

            # GFS数据
            d = pd.DataFrame()
            for day in date:
                for h in hour:
                    try:
                        df = pd.read_csv("/suncere/pyd/PycharmProjects/GFS_data/gfs.0p25.{}00.f{}.csv".format(day, h))
                    except Exception as result:
                        pass
                    else:
                        hf = df[df["StationCode"] == station]
                        d = pd.concat([d, hf], axis=0)
            d.index = range(len(d))
            d = pd.concat([datetime, d], axis=1)
            d = datetime1.merge(d, on="TimePoint", how="outer")
            d = d.interpolate(method="ffill")
            d = d.drop(labels=["StationCode", "PositionName"], axis=1)
            d.index = range(len(d))

            # 合并检测数据和GFS数据
            data = data.merge(d, on="TimePoint", how="outer")

            # 加入站点编号
            data["StationCode"] = [number] * len(data)

            # 划分滑动窗口
            input_seq = create_sequence(input_data=data,
                                        train_window=window1['train_window'],
                                        pollutant=pol['pollutant6'])

            # input_seq.iloc[:, 1:] = input_seq.iloc[:, 1:].interpolate(method="linear", limit_direction="backward")
            # input_seq.dropna(axis=0, how="any", inplace=True)
            input_seq.index = range(len(input_seq))

            single_city_data = single_city_data.append(input_seq)   # 一个城市中所有国控站点的数据

    # 如果某城市没有数据，则跳过该城市
    if single_city_data.empty:
        print("{}没有数据".format(city))
        continue

    """Xgboost模型训练"""
    # 创建日志
    time_str = "{}s".format(model_parameter['time_budget'])
    windows_str = "{}".format(window1['prediction_window'])
    filename = "Flaml_" + pol['pollutant6'] + "_" + windows_str + "_xgboost_" + time_str + "hourly"
    logging.basicConfig(
        filename="/suncere/pyd/PycharmProjects/output/all_city_train/{}/".format(city) + filename + ".log",
        filemode="w", level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"training_model:AutoML_{pol['pollutant6']}")
    logging.info(f"train_window:{window1['train_window']}, predict_window:{window1['prediction_window']}, "
                 f"window_step:{window1['window_step']}")

    all_start_time = time.time()

    single_city_data.iloc[:, 1:] = single_city_data.iloc[:, 1:].astype("float64")

    # 删除气象数据
    # single_city_data = single_city_data.drop(labels=["wind_spd", "wind_dir", "press", "tem", "hum"], axis=1)

    # 删除相关性相关特征
    single_city_data = single_city_data.drop(labels=["hour", "surface_hpbl", "surface_SUNSD", "isobaric_850_u",
                             "isobaric_700_v", "isobaric_500_u", "isobaric_500_v"], axis=1)

    # data = data.drop(labels=["TEM_MAX_24H", "TEM_MIN_24H", "PRS_MAX_24H", "PRS_MIN_24H", "spd_max_6h", "spd_max_12h"],
    #                  axis=1)

    single_city_data = single_city_data.drop(labels=["TimePoint"], axis=1)
    single_city_data.dropna(axis=0, how="any", inplace=True)

    single_city_data.index = range(len(single_city_data))

    # 保存特征
    logging.info(f"{pol['pollutant6']}的特征为：{single_city_data.columns[1:]}")
    logging.info("-" * 50)

    # 训练模型
    mse, mae, mar, r2 = PM10_flaml_automl.train_model(single_city_data, city)

    all_end_time = time.time()
    logging.info(f"总共花费的时间为：{all_end_time - all_start_time:.2f}s")
    print("总共花费的时间为：{:.2f}".format(all_end_time - all_start_time))