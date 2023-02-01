import copy
import pandas as pd
import numpy as np
from numpy.linalg import eig
from chinese_calendar import is_holiday
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cof import COF
from scipy.interpolate import interp1d, interp2d
from datetime import datetime
# from torch.nn import Embedding
# import torch
from config import *


def filter(args, var_origin):
    """ 平滑滤波 """
    df = copy.deepcopy(var_origin)
    df[pol['pollutant']] = savgol_filter(df[pol['pollutant']], args.filter_window, args.filter_rank, mode='nearest')
    return df


def loss_data_process(data):
    """缺失值处理"""
    data.dropna(axis=0, how="any", inplace=True)
    data = data.interpolate(method="linear")
    return data


def isolation_forest(data):
    """检测异常值，并用插值法替换异常值"""
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=100, random_state=rng, contamination=0.01)
    pollutant = data.iloc[:, 1: 7]
    clf.fit(pollutant)
    outliers_pred = clf.predict(pollutant)
    pollutant.iloc[outliers_pred == -1, :] = np.nan   # 将所有异常点中污染物的列的行赋值为nan
    # 线性插值法替换nan
    pollutant = pollutant.interpolate(method="linear")
    data.iloc[:, 1: 7] = pollutant
    # data.to_excel("./data/aoti")   # 保存到原数据中
    return data


# lof检测异常值
def lof_detection(data):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    for column1 in data.columns[: 6]:
        for column2 in data.columns[data.columns.get_loc(column1) + 1: 6]:
            data_new = data[[column1, column2]]  # 时间列和数据列
            lof_pred = lof.fit_predict(data_new)
            # data = data.drop(data_new[lof_pred == -1].index)
            data.loc[data_new[lof_pred == -1].index, [column1, column2]] = np.nan
            data.dropna(axis=0, how="any", inplace=True)
    return data


# cof检测异常值
def cof_detection(data):
    cof = COF(contamination=0.01, n_neighbors=20)
    for column1 in data.columns[1:]:
        for column2 in data.columns[data.columns.get_loc(column1):7]:
            data_new = data[[column1, column2]]  # 时间列和数据列
            cof_pred = cof.fit_predict(data_new)
            # data = data.drop(data_new[cof_pred == 1].index)
            data.loc[data_new[cof_pred == -1].index, [column1, column2]] = np.nan
            data = data.interpolate(method="linear")
    return data


# 3倍标准差方法
def var_detection(data):
    data_new = data.iloc[:, 1:7]   # 只针对污染物
    data_meaned = data_new - data_new.mean(axis=0)
    data_std = data_new.std(axis=0)
    for column in data_new.columns:
        for j in data_new.index:
            if (data_meaned[column][j] < -3*data_std[column]) | (data_meaned[column][j] > 3*data_std[column]):
                data_new[column][j] = np.nan
                data[column][j] = np.nan
    return data


def wind_spd_dir(data):
    """风速、风向处理"""
    data["wind_dir"] = data["wind_spd"] * np.sin(data["wind_dir"] * np.pi / 180)
    data["wind_spd"] = data["wind_spd"] * np.cos(data["wind_dir"] * np.pi / 180)
    # data = data.drop(labels=["wind_spd", "wind_dir"], axis=1)
    # data.to_excel("./data/aoti")
    return data


# Embedding层获取风向特征的信息
# def emb_layer(data):
#     wind_dir_tensor = torch.LongTensor(data["wind_dir"])   # 转换成tensor类型
#     embedding_dim = 3
#     emb = Embedding(num_embeddings=len(data), embedding_dim=embedding_dim)
#     embedding = emb(wind_dir_tensor)
#     emb_np = embedding.detach().numpy()   # 将tensor转换成numpy类型
#     emb_pd = pd.DataFrame(emb_np, columns=["wind_dir({})".format(i+1) for i in range(embedding_dim)])
#     data = data.join(emb_pd)
#     data = data.drop("wind_dir", axis=1)
#     return data


def wind_dir(data):
    """ 风向转换 """
    for i in data.index:
        if 0.0 <= data['wind_dir'][i] <= 45.0:
            data['wind_dir'][i] = 22.5
        elif 45.0 < data['wind_dir'][i] <= 90.0:
            data['wind_dir'][i] = 67.5
        elif 90.0 < data['wind_dir'][i] <= 135.0:
            data['wind_dir'][i] = 112.5
        elif 135.0 < data['wind_dir'][i] <= 180.0:
            data['wind_dir'][i] = 157.5
        elif 180.0 < data['wind_dir'][i] <= 225.0:
            data['wind_dir'][i] = 202.5
        elif 225.0 < data['wind_dir'][i] <= 270.0:
            data['wind_dir'][i] = 247.5
        elif 270.0 < data['wind_dir'][i] <= 315.0:
            data['wind_dir'][i] = 292.5
        elif 315.0 < data['wind_dir'][i] <= 360.0:
            data['wind_dir'][i] = 337.5
    return data


def onehot(data):
    """ 风向onehot编码 """
    data = wind_dir(data)
    data = pd.get_dummies(data, columns=['wind_dir'])
    return data


# 设置不同步长的训练窗口
# def create_sequence(input_data, train_window, prediction_window):
#     """取一定大小的训练窗口进行滑动"""
#     L = len(input_data)
#     ws_list = [24, 2, 3, 4]
#     input_seq = []; output_seq = []
#     input_seq_all = []; output_seq_all = []
#     for ws in ws_list:   # 考虑不同的窗口步长
#         for i in range(0, L - train_window - ws - prediction_window + 1, ws):
#             in_seq = input_data[i: i + train_window]   # 特征 [0,24]
#             out_seq = input_data.loc[i + train_window: i + train_window + prediction_window - 1, ["SO2"]]  # 一列为输出(标签) [24,48]
#             input_seq.append(in_seq)
#             output_seq.append(out_seq)
#         input_seq_all.append(input_seq)   # 将不同步长的滑动窗口全部存储在一个列表中
#         output_seq_all.append(output_seq)
#         input_seq = []
#         output_seq = []
#     return input_seq_all, output_seq_all   # input_seq_all[0]:步长为1时的窗口序列


# def create_sequence(input_data, train_window, prediction_window, ws, pollutant):
#     """输出单一污染物(pw=72, tw=1)"""
#     L = len(input_data)
#     in_seq_all = pd.DataFrame()
#     windows_quantity = 0   # 窗口数量
#     for i in range(0, L - train_window - ws - prediction_window + 1, ws):
#         in_seq = input_data[i: i + train_window]   # DataFrame
#         out_seq = input_data["{}".format(pollutant)][i + train_window: i + train_window + prediction_window]
#         # 将输出添加到窗口中
#         for i, j in enumerate(list(out_seq)):
#             in_seq["{}({})".format(pollutant, i)] = [j]*len(in_seq)
#         in_seq_all = in_seq_all.append(in_seq)
#         # in_seq_all = pd.DataFrame(in_seq_all.values)   # 修改索引值
#         windows_quantity += 1
#     return in_seq_all, windows_quantity


# def create_sequence(input_data, train_window, prediction_window, ws):
#     """输出六大污染物(预测下一小时的污染物数据)"""
#     L = len(input_data)
#     in_seq_all = pd.DataFrame()
#     windows_quantity = 0   # 窗口数量
#     for i in range(0, L - train_window - ws - prediction_window + 1, ws):
#         in_seq = input_data[i: i + train_window]   # DataFrame
#         for pollutant in args.pollutant_list:
#             out_seq = input_data["{}".format(pollutant)][i + train_window: i + train_window + prediction_window]
#             # 将输出添加到窗口中
#             for k, j in enumerate(list(out_seq)):
#                 in_seq["{}({})".format(pollutant, k)] = [j]*len(in_seq)
#         in_seq_all = in_seq_all.append(in_seq)
#             # in_seq_all = pd.DataFrame(in_seq_all.values)   # 修改索引值
#         windows_quantity += 1
#     return in_seq_all, windows_quantity


# def create_sequence(input_data, train_window, prediction_window, ws, poll):
#     """输出六大污染物(预测未来一天同时间的数据)"""
#     L = len(input_data)
#     in_seq_all = pd.DataFrame()
#     # windows_quantity = 0   # 窗口数量
#     for i in range(0, L - 24 + 1, ws):
#         in_seq = input_data[i: i + train_window]   # DataFrame
#         for pollutant in poll:
#             out_seq = input_data["{}".format(pollutant)][i + 24: i + 24 + prediction_window]
#             # 将输出添加到窗口中
#             for k, j in enumerate(list(out_seq)):
#                 in_seq["{}({})".format(pollutant, k)] = [j]*len(in_seq)
#         in_seq_all = in_seq_all.append(in_seq)
#             # in_seq_all = pd.DataFrame(in_seq_all.values)   # 修改索引值
#         # windows_quantity += 1
#     return in_seq_all


def create_sequence(input_data, train_window, pollutant):
    """alter windows(predict next hour)"""
    L = len(input_data)
    in_seq_all = pd.DataFrame()

    for i in range(train_window, L):
        in_seq = pd.DataFrame(input_data.iloc[i, 7:].values.reshape((1, len(input_data.columns[7:]))),
                              columns=input_data.columns[7:], index=[i])
        out_seq = input_data["{}".format(pollutant)][i]
        time_seq = input_data["TimePoint"][i]

        in_seq["{}".format(pollutant)] = out_seq
        in_seq["TimePoint"] = time_seq
        _1 = in_seq["{}".format(pollutant)]
        _2 = in_seq["TimePoint"]
        in_seq = in_seq.drop(labels=["{}".format(pollutant), "TimePoint"], axis=1)
        in_seq.insert(0, "{}".format(pollutant), _1)
        in_seq.insert(0, "TimePoint", _2)

        # 加入前一小时的污染物浓度
        for n, k in enumerate(range(i - train_window, i)):
            in_seq["{}({})".format(pollutant, n + 1)] = None
            in_seq["{}({})".format(pollutant, n + 1)] = input_data["{}".format(pollutant)][k]

        in_seq_all = in_seq_all.append(in_seq)
    in_seq_all.index = range(len(in_seq_all))
    return in_seq_all


# def create_sequence(input_data, train_window, prediction_window, ws, pollutant):
#     """alter windows(predict the same time's data of the next day)"""
#     L = len(input_data)
#     in_seq_all = pd.DataFrame()
#
#     for i in range(train_window, L):
#         in_seq = pd.DataFrame(input_data.iloc[i, 9:].values.reshape((1, len(input_data.columns[9:]))),
#                               columns=input_data.columns[9:], index=[i])
#         out_seq = input_data["{}".format(pollutant)][i]
#         time_seq = input_data["TimePoint"][i]
#
#         in_seq["{}".format(pollutant)] = out_seq
#         in_seq["TimePoint"] = time_seq
#         _1 = in_seq["{}".format(pollutant)]
#         _2 = in_seq["TimePoint"]
#         in_seq = in_seq.drop(labels=["{}".format(pollutant), "TimePoint"], axis=1)
#         in_seq.insert(0, "{}".format(pollutant), _1)
#         in_seq.insert(0, "TimePoint", _2)
#
#         if i < 24:
#             in_seq["{}({})".format(pollutant, 1)] = input_data["{}".format(pollutant)][i]
#         else:
#             in_seq["{}({})".format(pollutant, 1)] = input_data["{}".format(pollutant)][i-24]
#
#         in_seq_all = in_seq_all.append(in_seq)
#     return in_seq_all


def hour_data(data):
    """小时特征"""
    data["hour"] = data["TimePoint"].dt.hour
    return data


def time_feature(data):
    """季节、月份特征"""
    data["TimePoint"] = pd.to_datetime(data["TimePoint"])
    data["month"] = data["TimePoint"].dt.month.astype(float)
    # 不能直接用quarter，需自定义季节[3,4,5]月为春天...
    # data["quarter"] = data["TimePoint"].dt.quarter
    data["quarter"] = data["TimePoint"].dt.quarter
    for index in data.index:
        if data["month"][index] in [3, 4, 5]:
            data["quarter"][index] = 1.0
            continue
        if data["month"][index] in [6, 7, 8]:
            data["quarter"][index] = 2.0
            continue
        if data["quarter"][index] in [9, 10, 11]:
            data["quarter"][index] = 3.0
            continue
        if data["quarter"][index] in [1, 2, 12]:
            data["quarter"][index] = 4.0
            continue
        data["quarter"] = data["quarter"].astype(float)
    return data


def vocation_feature(data):
    """节假日特征"""
    data["vocation"] = data["TimePoint"].map(lambda x: is_holiday(x))
    for i in data.index:
        if data["vocation"][i] is True:
            data["vocation"][i] = 1
        else:
            data["vocation"][i] = 0
    return data


def dayofweek_feature(data):
    """星期特征"""
    data["weekday"] = data["TimePoint"].dt.dayofweek
    return data


def add_lastyear_data(data, pollutant):
    """添加上一年同时间段的数据"""
    data["TimePoint"] = pd.to_datetime(data["TimePoint"])
    data["year"] = data["TimePoint"].dt.year
    data["{}_lastyear".format(pollutant)] = 0
    # 删除2020.2.29的数据
    date1 = data["TimePoint"] >= "2020-02-29 00:00:00"
    date2 = data["TimePoint"] <= "2020-02-29 23:00:00"
    data = data[~data["TimePoint"].isin(list(data[date1&date2]["TimePoint"]))]
    data.index = range(len(data))

    for i in data.index:
        # 没有上一年，直接用当年的的数据
        if data["year"][i] == 2019:
            data["{}_lastyear".format(pollutant)][i] = data[pollutant][i]
        if (data["year"][i] == 2020) | (data["year"][i] == 2021) | (data["year"][i] == 2022):
            if i < 365 * 24:
                data["{}_lastyear".format(pollutant)][i] = data[pollutant][i]
            else:
                if i - 365 * 24 not in data.index:
                    data["{}_lastyear".format(pollutant)][i] = data[pollutant][i]
                else:
                    data["{}_lastyear".format(pollutant)][i] = data[pollutant][i - 365 * 24]
        else:
            continue

    # data = data.drop(["year"], axis=1)
    # _ = data["{}_lastyear".format(pollutant)]
    # data = data.drop("{}_lastyear".format(pollutant), axis=1)
    # data.insert(7, "{}_lastyear".format(pollutant), _)
    # data.index = range(len(data))
    return data


def add_lastday_data(data, pollutant):
    """添加前一天相同时间的数据"""
    data["{}_lastday".format(pollutant)] = 0
    for i in data.index:
        if i < 24:   # 下标从0开始
            data["{}_lastday".format(pollutant)][i] = data[pollutant][i]
        else:
            if i - 24 not in data.index:
                data["{}_lastday".format(pollutant)][i] = data[pollutant][i]
            else:
                data["{}_lastday".format(pollutant)][i] = data[pollutant][i - 24]
    # _ = data["{}_lastday".format(pollutant)]
    # data = data.drop("{}_lastday".format(pollutant), axis=1)
    # data.insert(7, "{}_lastday".format(pollutant), _)
    return data


def add_lastweek_data(data):
    """添加前一周相同时间的数据"""
    data["{}_lastweek".format(args.pollutant)] = 0
    for i in data.index:
        if i < 24*7:   # 下标从0开始
            data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i]
        else:
            data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i - 24]
    return data


def add_lastmonth_data(data):
    data["month"] = data["TimePoint"].dt.month
    for i in data.index:
        if data["month"][i] in [1, 2, 4, 6, 8, 9, 11]:
            if i < 24*31:
                data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i]
            else:
                data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i - 24*31]
        elif data["month"][i] in [5, 7, 10, 12]:
            if i < 24*30:
                data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i]
            else:
                data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i - 24 * 30]
        else:
            if i < 24*28:
                data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i]
            else:
                data["{}_lastweek".format(args.pollutant)][i] = data[args.pollutant][i - 24 * 28]
    return data


def uniform_scale(data):
    """归一化"""
    transfer = MinMaxScaler()
    data = transfer.fit_transform(data)
    return data


def pca(data):
    """主成分分析"""
    data = data - data.mean(axis=0)   # 中心化
    data_cov = np.cov(data.T, ddof=0)   # 协方差
    feature_value, feature_vector = eig(data_cov)   # 协方差矩阵的特征值和特征向量
    feature_sum = sum(feature_value)
    explain_var = [(i/feature_sum) for i in sorted(feature_value, reverse=True)]   # 解释方差
    cumsum_var = np.cumsum(explain_var)   # 累计解释方差
    return cumsum_var


def space_feature(data):
    """引入空间特征"""
    pass


def max_min_tem(data):
    """加入一天内最高和最低气温"""
    data["TEM_MAX_24H"] = 0
    data["TEM_MIN_24H"] = 0

    for i in range(0, len(data), 24):
        data["TEM_MAX_24H"][i:i + 24] = data["tem"][i:i + 24].max()
        data["TEM_MIN_24H"][i:i + 24] = data["tem"][i:i + 24].min()
    if len(data) % 24 != 0:
        for j in np.arange(int(len(data)/24)*24, len(data)):
            data["TEM_MAX_24H"][j] = data["tem"][j]
            data["TEM_MIN_24H"][j] = data["tem"][j]
    # _1 = data["TEM_MAX_24H"]
    # _2 = data["TEM_MIN_24H"]
    # data = data.drop(labels=["TEM_MAX_24H", "TEM_MIN_24H"], axis=1)
    # data.insert(7, "TEM_MAX_24H", _1)
    # data.insert(7, "TEM_MIN_24H", _2)
    return data



def max_min_press(data):
    """加入一天内最高和最低气压"""
    data["PRS_MAX_24H"] = 0
    data["PRS_MIN_24H"] = 0
    for i in range(0, len(data), 24):
        data["PRS_MAX_24H"][i:i + 24] = data["press"][i:i + 24].max()
        data["PRS_MIN_24H"][i:i + 24] = data["press"][i:i + 24].min()
    if len(data) % 24 != 0:
        for j in np.arange(int(len(data) / 24) * 24, len(data)):
            data["PRS_MAX_24H"][j] = data["press"][j]
            data["PRS_MIN_24H"][j] = data["press"][j]
    # _1 = data["PRS_MAX_24H"]
    # _2 = data["PRS_MIN_24H"]
    # data = data.drop(labels=["PRS_MAX_24H", "PRS_MIN_24H"], axis=1)
    # data.insert(7, "PRS_MAX_24H", _1)
    # data.insert(7, "PRS_MIN_24H", _2)
    return data

def press_change(data):
    """前一天的变压"""
    data["press_change"] = 0
    data["press_change"][0:24] = data["press"][0:24].max() - data["press"][0:24].min()
    for i in range(24, len(data), 24):
        data["press_change"][i: i+24] = data["press"][i-24: i].max() - data["press"][i-24: i].min()
    if len(data) % 24 != 0:
        data["press_change"][int((len(data) / 24) * 24):] = \
            data["press"][int((len(data) / 24) * 24) - 24: int((len(data) / 24) * 24)].max() - \
            data["press"][int((len(data) / 24) * 24): int((len(data) / 24) * 24)].min()
    return data


def max_spd_6h_12h(data):
    """加入6小时和12小时的极大风速"""
    data["spd_max_6h"] = 0
    data["spd_max_12h"] = 0
    for i in range(0, len(data), 6):
        data["spd_max_6h"][i:i + 6] = data["wind_spd"][i:i + 6].max()
    for i in range(0, len(data), 12):
        data["spd_max_12h"][i:i + 12] = data["wind_spd"][i:i + 12].max()

    if len(data) % 6 != 0:
        for j in np.arange(int(len(data) / 6) * 6, len(data)):
            data["spd_max_6h"][j] = data["wind_spd"][j]
    if len(data) % 12 != 0:
        for j in np.arange(int(len(data) / 12) * 12, len(data)):
            data["spd_max_12h"][j] = data["wind_spd"][j]
    return data

