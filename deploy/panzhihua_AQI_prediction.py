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
import copy


warnings.filterwarnings("ignore")


def hourly_predict_data(pred_data, hourly_data, pollutant):
    SO2_mae = [6.262651117748454, 8.370364481797978, 9.512259555170518, 10.274077306184372, 10.641478233053258,
               10.73894886591593, 10.8904089544033, 10.794516085388475, 10.98884743623352, 10.98501516352665,
               11.092038765868514, 11.241119755939685, 11.29103289464533, 11.181589072647476, 11.29824649311536,
               11.086328835887384, 11.05999286384213, 11.114012456305607, 11.28014456420859, 11.103662165248489,
               10.960217293386942, 10.902824468575542, 10.79280798129583, 10.76179747557954, 10.778676057976815,
               10.78723017832931, 10.959133209392888, 10.995233090661085, 11.128347796177753, 11.041505895477421,
               11.105987394960934, 11.191125360390217, 11.077855682053473, 11.000370448209855, 10.924627170316056,
               11.205056405139395, 11.193770114938674, 11.136785698488323, 11.218267859601516, 11.181768100216368,
               11.214803457319764, 11.176411884239496, 11.31112532922102, 11.109543378342629, 11.080204450205397,
               10.94497703419099, 10.829755106399409, 10.779664142889242, 10.873823229760665, 11.098868513781106,
               11.04099700850211, 11.00377603355008, 10.874959537491222, 10.898997552729604, 10.915614528837455,
               10.913788335737411, 10.819787099717225, 10.880910729806876, 11.208481587887745, 11.065243004385884,
               11.083327194978887, 11.222983672797234, 11.34107691060505, 11.574225116092402, 11.713779551180389,
               11.49400495526731, 11.338010626452217, 11.235012064378312, 11.269389532870958, 11.208247528783343,
               10.98793371246828, 11.070069137161973]
    NO2_mae = [7.9278700047383595, 10.012868813167234, 11.044540351427115, 11.597011163699873, 11.985012289194966,
               12.319499604523125, 12.34605852335682, 12.33221253432817, 12.26250761966961, 12.446513673724587,
               12.33168493343026, 12.40595343857359, 12.60015480040095, 12.779435643224987, 12.81841463398629,
               12.692253543164037, 12.664045957354185, 12.523401002371703, 12.340078828812048, 12.276731670751367,
               12.457500954184512, 12.457025531111883, 12.138070726457045, 12.035977990227032, 12.270779859531375,
               12.515064444675268, 12.63907291021807, 12.58310792946018, 12.547718292058011, 12.7690370235904,
               12.627565360900986, 12.503653916020529, 12.348953688355756, 12.438168048904933, 12.233768277361987,
               12.346560499736299, 12.579620718748192, 12.78098580254952, 12.914414317900965, 12.849504158401007,
               12.850909560695534, 12.736583207221903, 12.535040309171807, 12.654730022381514, 12.792098071197044,
               12.791443199401442, 12.60570434580275, 12.37882405885453, 12.468380406435953, 12.577628910342035,
               12.640578085398312, 12.743829169201426, 12.700622716100456, 12.544245429189443, 12.701950356701929,
               12.635110200390077, 12.5911510757687, 12.676759290577452, 12.487992385830244, 12.42209435230093,
               12.669179488445014, 12.685534086481006, 12.576173449462258, 12.589725857960394, 12.568312401994442,
               12.463822824362023, 12.425239435207796, 12.53179976005047, 12.993126918757103, 12.948871182551096,
               12.858809949749695, 12.649272424281868]
    CO_mae = [0.3048766897905033, 0.40231751000703586, 0.4526159671577006, 0.49939811367136666, 0.5118462490044742,
              0.5441740467417013, 0.5376891050573629, 0.5496458722177887, 0.5517284108766677, 0.5663991679162875,
              0.5699635193824077, 0.572833391788015, 0.5832247585017888, 0.5708211430221826, 0.5738747164659677,
              0.5694786456088091, 0.5738157684456594, 0.5642101745785263, 0.5708860995111453, 0.5721461464247332,
              0.566468338691172, 0.5678715721346563, 0.5542242581595096, 0.5465957618818192, 0.5454011673449818,
              0.5625452606884143, 0.567149681874752, 0.5587561816480653, 0.5633657409311025, 0.5796414133585288,
              0.5803600142899119, 0.5795668606568861, 0.5792576808582929, 0.5885247338305927, 0.595560977771228,
              0.5901782077353228, 0.6016862308313016, 0.5893526743983382, 0.5892374114104155, 0.5812511625476234,
              0.5872337571543345, 0.5864287093852182, 0.5827844831514254, 0.5823515204268396, 0.5975587692460812,
              0.5926206323843806, 0.5847354714062162, 0.5646834368982987, 0.5635486166817759, 0.5620573250302795,
              0.5761681203630538, 0.5763159710682499, 0.5808253755218293, 0.5822706321225024, 0.5864870871789392,
              0.5920135155772819, 0.5985174263194434, 0.6087089259749799, 0.6195311982208558, 0.5949587275687909,
              0.5985468766809485, 0.5965037605187336, 0.5860186292043768, 0.5852885487203918, 0.5853899469333259,
              0.5949767656975355, 0.5929352658279748, 0.5852438273217909, 0.5821389371481923, 0.5791538048187048,
              0.5771205572984428, 0.5677100982937539]
    O3_mae = [12.424318114904336, 15.141077079834623, 16.941284133610996, 17.82430615499677, 18.696640091781497,
              19.36685458328872, 20.046555301151205, 20.202800969247743, 20.03049449247868, 20.32169355017565,
              20.18888298348074, 20.483460960195693, 20.798157919142355, 20.53828811122977, 20.5280018542688,
              20.420793243055364, 20.333846829975208, 20.225946381974634, 19.869363780267424, 19.635534916746366,
              19.83474466705098, 19.706487084507778, 19.18077917372412, 18.814652933714648, 19.370892316559214,
              19.96628771377689, 20.27282479562223, 20.08179713022889, 20.09092163652386, 20.589227904436193,
              20.612112025246535, 20.45299811220627, 20.521054208242685, 20.54783641269668, 20.701890540862117,
              20.90305240051573, 20.85170164676033, 20.803236174380665, 20.619114498117497, 20.55559506579845,
              20.60163551283471, 20.438312223170193, 20.37549893550585, 20.463399191394217, 20.505503085312274,
              20.443416380793494, 20.70687962381975, 20.311649787937593, 20.630343376644074, 20.898376928598957,
              20.994736865217632, 20.58828434387609, 20.543654281280563, 20.894145366445187, 20.956651798900893,
              20.804579511543782, 21.097140016600637, 20.90638625428837, 20.855723955471, 21.10452381835226,
              21.158658397344894, 21.040596963832737, 20.93067613752127, 20.946734346067952, 20.779086952661736,
              20.678858060276138, 20.43857765612078, 20.37259714791524, 20.710634506194936, 20.489978645720154,
              20.678325839467828, 20.48157897690196]
    PM2_5_mae = [6.455965637711751, 7.661583100090226, 8.667613878130487, 9.345598169530493, 9.773988643574196,
                 10.013218838962699, 10.210040741104336, 10.403714224110937, 10.608603745721245, 10.753094912569168,
                 10.825436206322395, 10.811897964865667, 10.777946003313636, 10.709707325579242, 10.870983771568127,
                 10.901263143924705, 10.972088944198523, 10.968789815355821, 10.977421368318907, 11.00921113989149,
                 10.956492869484302, 11.020219301898406, 10.978315668270932, 10.965835644456746, 11.00477574120024,
                 11.03710656689558, 11.137813126797258, 11.060841313529219, 11.20685935178525, 11.226386837660662,
                 11.221359878899428, 11.066137288957414, 11.0956653615329, 11.076366635220028, 11.161297738494746,
                 11.19836552787341, 11.33867723603556, 11.17735049868953, 11.259167054447003, 11.213109452430704,
                 11.290370683162656, 11.285414269471412, 11.148094122568086, 11.245030110770244, 11.165481370401817,
                 11.138320536969232, 11.13214271541417, 11.1041624932455, 11.060559310237528, 11.249353436747532,
                 11.318419913360463, 11.331121825851323, 11.489907897107138, 11.187079592005356, 11.123892262023652,
                 11.108076189480144, 11.165600644130715, 11.233843954805682, 11.18215373912331, 11.272905980239768,
                 11.321800269566975, 11.339029502445408, 11.38426953815918, 11.395509418701346, 11.37949998895757,
                 11.426426053004203, 11.351068453228566, 11.262919188683714, 11.277477989996585, 11.300500814460491,
                 11.341551236011508, 11.465481917417819]
    PM10_mae = [10.667834863966773, 9.609767822705757, 10.816215992372971, 14.166022717461484, 14.491368779885493,
                15.008122007107541, 15.308866601069898, 15.515434038225944, 15.552685702021892, 15.549825854691072,
                15.571875154727108, 15.722123200547243, 15.804270961544029, 15.717177536215326, 15.851253606723054,
                15.870756846069998, 15.893855327630831, 16.253601092359638, 16.009629456544644, 16.016342313504715,
                16.206503646695346, 16.13715879416262, 16.097674081659292, 15.90554301499904, 16.11358893371023,
                16.239847548196344, 16.226032409770117, 16.499615478609243, 16.323339607522712, 16.50398239740684,
                16.639210926836363, 16.34391987344167, 16.15049117565243, 16.407977621306003, 16.376999744603122,
                16.346420898316172, 16.360598242897158, 16.357571901010925, 16.53177265408631, 16.43766313999785,
                16.551078943974655, 16.592971813445047, 16.527889713805735, 14.606835480472885, 14.697354959552161,
                14.1211958733072, 16.49805867859718, 16.500064483379468, 16.577616063987477, 16.368811526594005,
                16.327654571001815, 16.421514521892995, 16.421261498715356, 16.4822508044309, 16.400914893981987,
                16.55662058992114, 16.35490987154498, 16.528647964840864, 16.588576315017633, 16.63495781577224,
                16.375726776230398, 16.31466083559687, 16.25411692913485, 14.529459356993137, 16.2492200978973,
                16.282027316755492, 14.663703915372738, 14.669741795191516, 14.873683300277555, 14.70414419883672,
                14.675493500863784, 14.398047964047374]

    # 前三天
    for i in range(72):
        # 加载模型
        model = pickle.load(
            open(
                "/usr/xgboost/models/sichuan_model/pred72(+GFS)/panzhihua/{}/AutoML_{}_{}(h_gfs_panzhihua_pred72).dat".format(
                    pollutant,
                    pollutant,
                    i+1),
                "rb"))
        if i > 24:
            pred_data["{}_lastday".format(pollutant)][i+1] = pred_data["{}".format(pollutant)][i-24+1]
            hourly_data["{}_lastday".format(pollutant)][i + 1] = hourly_data["{}".format(pollutant)][i - 24 + 1]

        # 预测每一行的值，赋值给下一列
        x = pred_data[i:i+2]

        # 预测
        if pollutant == "SO2":
            y_pred = model.predict(x)  # 单行不能预测
            pred_data["{}".format(pollutant)][i + 2] = y_pred[-1] + SO2_mae[i]
            hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1] + SO2_mae[i]
        if pollutant == "NO2":
            y_pred = model.predict(x)  # 单行不能预测
            pred_data["{}".format(pollutant)][i + 2] = y_pred[-1] + NO2_mae[i]
            hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1] + NO2_mae[i]
        if pollutant == "CO":
            y_pred = model.predict(x)  # 单行不能预测
            pred_data["{}".format(pollutant)][i + 2] = y_pred[-1] + CO_mae[i]
            hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1] + CO_mae[i]
        if pollutant == "O3":
            y_pred = model.predict(x)  # 单行不能预测
            pred_data["{}".format(pollutant)][i + 2] = y_pred[-1] + O3_mae[i]
            hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1] + O3_mae[i]
        if pollutant == "PM2.5":
            y_pred = model.predict(x)  # 单行不能预测
            pred_data["{}".format(pollutant)][i + 2] = y_pred[-1] + PM2_5_mae[i]
            hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1] + PM2_5_mae[i]
        if pollutant == "PM10":
            y_pred = model.predict(x)  # 单行不能预测
            pred_data["{}".format(pollutant)][i + 2] = y_pred[-1] + PM10_mae[i]
            hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1] + PM10_mae[i]

    # 第四~十天
    for i in np.arange(72, 240, 1):
        # 加载模型
        model = pickle.load(
            open(
                "/usr/xgboost/models/sichuan_model/pred72(+GFS)/panzhihua/{}/AutoML_{}_{}(h_gfs_panzhihua_pred72(4d_10d)).dat".format(
                    pollutant,
                    pollutant,
                    i - 71),
                "rb"))
        if i > 24:
            pred_data["{}_lastday".format(pollutant)][i + 1] = pred_data["{}".format(pollutant)][i - 24 + 1]
            hourly_data["{}_lastday".format(pollutant)][i + 1] = hourly_data["{}".format(pollutant)][i - 24 + 1]

        # 预测每一行的值，赋值给下一列
        x = pred_data[i:i + 2]


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


def vocation_feature(data):
    """节假日特征"""
    # data["vocation"] = data["TimePoint"].map(lambda x: is_holiday(x))
    # for i in data.index:
    #     if data["vocation"][i] is True:
    #         data["vocation"][i] = 1
    #     else:
    #         data["vocation"][i] = 0
    data["vocation"] = 0
    for i in data.index:
        if data["weekday"][i] in [6, 7]:
            data["vocation"][i] = 1
        else:
            data["vocation"][i] = 0
    return data


def dayofweek_feature(data):
    """星期特征"""
    data["weekday"] = data["TimePoint"].dt.dayofweek
    return data


def feature_select(pred_data, pollutant):
    """特征筛选"""
    if pollutant == "SO2":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press',
                          'tem', 'hum', 'press_change', 'surface_sp', 'surface_SUNSD', 'surface_t', 'surface_gust',
                          'surface_prate', 'quarter', 'month', 'hour', 'SO2_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "NO2":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press',
                          'tem', 'hum', 'press_change', 'surface_sp', 'surface_SUNSD', 'surface_t', 'surface_gust',
                          'surface_prate', 'quarter', 'month', 'hour', 'NO2_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "CO":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press',
                          'tem', 'hum', 'press_change', 'surface_sp', 'surface_SUNSD', 'surface_t', 'surface_gust',
                          'surface_prate', 'quarter', 'month', 'hour', 'CO_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "O3":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press',
                          'tem', 'hum', 'TEM_MAX_24H', 'TEM_MIN_24H', 'surface_SUNSD', 'surface_t', 'highCloudLayer_hcc',
                          'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'quarter', 'month', 'hour', 'vocation', 'weekday',
                          'O3_lastyear', 'O3_lastday', 'StationCode', 'press_change']]

    if pollutant == "PM2.5":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press',
                          'tem', 'hum', 'TEM_MAX_24H', 'TEM_MIN_24H', 'press_change', 'surface_hpbl', 'surface_t',
                          'surface_prate', 'surface_gust', 'isobaric_500_gh', 'isobaric_500_u', 'isobaric_500_v',
                          'quarter', 'month', 'hour', 'vocation', 'weekday', 'PM2.5_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "PM10":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press',
                          'tem', 'hum', 'TEM_MAX_24H', 'TEM_MIN_24H', 'press_change', 'surface_hpbl', 'surface_t',
                          'surface_prate', 'surface_gust', 'isobaric_500_gh', 'isobaric_500_u', 'isobaric_500_v',
                          'hour', 'vocation', 'weekday', 'PM10_lastday', 'O3_lastday', 'StationCode']]

    data.index = range(len(data))

    return data


def wind_spd_dir(data):
    """风速、风向处理"""
    data["wind_dir"] = data["wind_spd"] * np.sin(data["wind_dir"] * np.pi / 180)
    data["wind_spd"] = data["wind_spd"] * np.cos(data["wind_dir"] * np.pi / 180)
    # data = data.drop(labels=["wind_spd", "wind_dir"], axis=1)
    return data


def get_weather_data(data):
    """加入气象数据 -- 四川pred72的城市需要"""
    weather_data = pd.read_csv(
        "/mnt/real/WEATHER/WEATHER_{}.csv".format(
            (pd.to_datetime(pd.datetime.now()) + timedelta(days=-1)).strftime("%Y-%m-%d")))
    data["tem"] = weather_data["TEM"][(weather_data["CITY"] == "攀枝花市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press'] = weather_data['PRS'][(weather_data["CITY"] == "攀枝花市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['hum'] = weather_data['RHU'][(weather_data["CITY"] == "攀枝花市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press_change'] = weather_data['PRS_CHANGE_24H'][(weather_data["CITY"] == "攀枝花市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_spd'] = weather_data['WIN_S_INST_MAX'][(weather_data["CITY"] == "攀枝花市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_dir'] = weather_data['WIN_D_INST_MAX'][(weather_data["CITY"] == "攀枝花市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data = wind_spd_dir(data)
    return data


def get_lastyear_data():
    now_data = pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d"))
    d_list = []
    for i in range(11):
        try:
            d = pd.read_csv("/mnt/real/AIR/Station_{}-{}-{}.tsv".format(
                now_data.year - 1, str(now_data.month).rjust(2, "0"), str((now_data + timedelta(days=i-1)).day).rjust(2, "0")), sep="\t")
        except:
            d = pd.read_csv("/mnt/real/AIR/Station_{}-{}-{}.csv".format(
                now_data.year - 1, str(now_data.month).rjust(2, "0"), str((now_data + timedelta(days=i-1)).day).rjust(2, "0")), sep="\t")
        finally:
            d_list.append(d)

    return d_list


def get_data():
    # 获取时间列：前一晚23:00~十天后00:00
    now_data = pd.datetime.now().strftime("%Y-%m-%d %H")
    now_data = pd.to_datetime(now_data)
    timepoint = [now_data + timedelta(hours=-6) + timedelta(hours=+i) for i in range(242)]
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

    # 只取宜宾市的数据
    pred_data = pred_data[pred_data["Area"] == "攀枝花市"]

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
    filename = "output_panzhihua"
    logging.basicConfig(filename="/usr/xgboost/deploy/" + filename + ".log",
                        filemode="a", level=logging.INFO,
                        format="%(message)s")
    # 读取城市编号
    citys = pd.read_csv("/usr/xgboost/deploy/citys.csv")

    # O3上一年同时段的数据
    d_list = get_lastyear_data()
    # d_list = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11]
    # 按照时间进行排序
    for d in d_list:
        d.rename(columns={"O3-1h-24h": "O3"}, inplace=True)
        d = d[["Hour", "StationCode", "O3"]]
        # d.sort_values(by="Hour", inplace=True)

    pollutant_list = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]

    timepoint, data, la = get_data()   # 监测数据
    gfs_data = concat_gfs.all_timepoint   # GFS(已经分组)
    for (code, group), (code1, group1) in zip(data, la):
        # try:
        group.index = range(len(group))

        t1 = time.time()

        group = group.drop(labels=["hour"], axis=1)
        hourly_data = timepoint.merge(group, on="TimePoint", how="outer")   # 还差GFS
        # GFS数据加上时间
        try:
            single_station_gfs = gfs_data.get_group(code)
            single_station_gfs.index = range(len(single_station_gfs))
            sole_station = pd.concat([concat_gfs.d1, single_station_gfs], axis=1)
            sole_station = concat_gfs.d2.merge(sole_station, how="outer", on="TimePoint")
            sole_station = sole_station.interpolate(method="ffill")
            sole_station = sole_station.interpolate(method="bfill")
            sole_station = sole_station.drop(labels=["Area", "StationCode", "UniqueCode"], axis=1)

            sole_station = sole_station.loc[3:, :]
        except:
            continue
        else:

            # 合并监测数据和GFS数据
            hourly_data = hourly_data.merge(sole_station, on="TimePoint", how="outer")
            hourly_data.index = range(len(hourly_data))

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
                    hourly_data["{}_lastday".format(p)][2:2 + 24] = lastday
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
            hourly_data["O3_lastyear"][hourly_data["O3_lastyear"] == "—"] = np.nan
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

            # 加入一天内的温度最大值和最小值
            hourly_data["TEM_MAX_24H"], hourly_data["TEM_MIN_24H"] = 0, 0
            hourly_data["TEM_MAX_24H"][:2] = hourly_data["surface_t"][:2].max()
            hourly_data["TEM_MIN_24H"][:2] = hourly_data["surface_t"][:2].min()
            for i in np.arange(2, 219, 24):
                hourly_data["TEM_MAX_24H"][i: i + 24] = hourly_data["surface_t"][i: i + 24].max()
                hourly_data["TEM_MIN_24H"][i: i + 24] = hourly_data["surface_t"][i: i + 24].min()

            # 加入假期和工作日
            hourly_data = dayofweek_feature(hourly_data)
            hourly_data = vocation_feature(hourly_data)

            # 获取气象数据
            hourly_data = get_weather_data(hourly_data)

            # hourly_data.rename(columns={"SO2": "SO2(1)", "NO2": "NO2(1)", "CO": "CO(1)", "O3": "O3(1)", "PM2.5": "PM2.5(1)", "PM10": "PM10(1)"}, inplace=True)

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
                hourly_predict_data(pollutant_data, hourly_data, pollutant)   # 预测
                hourly_data["{}".format(pollutant)] = carry_ahead(hourly_data, "{}".format(pollutant))   # 四舍六入五成双
                # hourly_data.rename(columns={"{}(1)".format(pollutant): "{}".format(pollutant)}, inplace=True)
                logging.info(f"{pollutant}小时值预测结果:{list(hourly_data[pollutant])}")

            hourly_predict_pol = hourly_data[["TimePoint", "UniqueCode", "PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]][2:]
            hourly_predict_pol["TimePoint"] = hourly_predict_pol["TimePoint"].apply(lambda x: x.strftime("%Y%m%d%H"))
            hourly_predict_pol["UniqueCode"] = hourly_predict_pol["UniqueCode"].astype(int)
            hourly_predict_pol.index = range(len(hourly_predict_pol))
            try:
                hourly_predict_pol.to_csv(
                    "/mnt/output/AQMS/station/AQMS_hourly_{}.txt".format(pd.datetime.now().strftime("%Y%m%d")),
                    sep=",", mode="a", header=0, index=0)
            except:
                pass

            # 污染物日均值
            daily_pred = pd.DataFrame(columns=hourly_predict_pol.columns, index=range(10))
            daily_pred["TimePoint"] = [pd.to_datetime(pd.datetime.now().strftime("%Y%m%d")) + timedelta(days=+i) for i
                                       in range(10)]
            daily_pred["TimePoint"] = daily_pred["TimePoint"].apply(lambda x: x.strftime("%Y%m%d"))
            daily_pred["UniqueCode"] = [group["UniqueCode"][0]] * 10

            for pollutant in hourly_predict_pol.columns[2:]:

                # 计算浓度日均值
                daily_pred = daily_pred_data(hourly_predict_pol, daily_pred, pollutant)

                # 四舍六入五成双
                # 日均值
                daily_pred[pollutant] = carry_ahead(daily_pred, pollutant)
                logging.info(f"{pollutant}日均值预测结果:{list(daily_pred[pollutant])}")
            daily_pred.rename(columns={"O3": "O38h"})
            try:
                daily_pred.to_csv(
                    "/mnt/output/AQMS/station/AQMS_daily_{}.txt".format(pd.datetime.now().strftime("%Y%m%d")),
                    sep=",", mode="a", header=0, index=0)
            except:
                pass

            # 城市结果保存
            hourly_predict_pol_city = copy.deepcopy(hourly_predict_pol)
            daily_pred_city = copy.deepcopy(daily_pred)

            citys["bool"] = [citys["名称"][i] in group["Area"][0] for i in range(len(citys))]
            hourly_predict_pol_city["UniqueCode"] = list(citys["城市编码"][citys["bool"] == True]) * len(hourly_predict_pol_city)
            daily_pred_city["UniqueCode"] = list(citys["城市编码"][citys["bool"] == True]) * len(daily_pred)

            hourly_predict_pol_city.to_csv(
                "/mnt/output/AQMS/city/AQMS_hourly_{}.txt".format(pd.datetime.now().strftime("%Y%m%d")),
                sep=",", mode="a", header=0, index=0)
            daily_pred_city.to_csv(
                "/mnt/output/AQMS/city/AQMS_daily_{}.txt".format(pd.datetime.now().strftime("%Y%m%d")),
                sep=",", mode="a", header=0, index=0)

            t2 = time.time()
            print("单个城市的预测时间：{}s".format(t2-t1))
        # except:
        #     continue

    end_time = time.time()
    logging.info(f"所花费的时间：{end_time - start_time}s")
    print("花费时间：{}s".format(end_time - start_time))



