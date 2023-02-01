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

pd.set_option("display.max_rows", None)

warnings.filterwarnings("ignore")


def hourly_predict_data(pred_data, hourly_data, pollutant):
    SO2_mae = [1.6088642525069068, 2.0324820258376928, 2.1699448772121404, 2.2149935160951664, 2.311808972762771,
               2.3374238497749302, 2.3369908372448434, 2.3584576168825255, 2.316723738064941, 2.3704357186488507,
               2.390024379322564, 2.3602992774622216, 2.3091121747281096, 2.3766272067967598, 2.3670350003682623,
               2.4236130654813963, 2.407015636103867, 2.313669992189347, 2.417195915472473, 2.4164069864362485,
               2.3870314943574362, 2.398320169493459, 2.433816948716674, 2.38493085359665, 2.40766361367488,
               2.491828054530091, 2.476435206022032, 2.4398330002865287, 2.4738356010604114, 2.440343089861942,
               2.43277722569501, 2.4460869567012375, 2.441532400752971, 2.46670900604782, 2.4780212771624814,
               2.4884569384499358, 2.494347282101168, 2.4885216638933154, 2.471868189920578, 2.514896650053595,
               2.5958880768881256, 2.6079672153522786, 2.5332158140675074, 2.5535824625178263, 2.526156027956602,
               2.491076459054681, 2.5106344707366772, 2.487779878207378, 2.4860801826145256, 2.5306299513098245,
               2.574724332625139, 2.5659989283232374, 2.5962615279108427, 2.5999276963124602, 2.5352762490353506,
               2.5368263204205803, 2.5813518104202844, 2.5524541331788093, 2.642596108265325, 2.632151626693176,
               2.562312974523348, 2.52895115679776, 2.495912017031689, 2.557308084833101, 2.61798717989859,
               2.683428111539955, 2.6144122492210733, 2.6441953050110594, 2.6153647312627504, 2.566653796245478,
               2.530733942028159, 2.5711032386878436]
    NO2_mae = [3.9278190621061664, 5.461190045897019, 6.242724376013907, 8.31659201117668, 7.073676002506068,
               8.6330156726696, 8.864970963010354, 8.855521720141404, 8.858467676360341, 8.97007185965328,
               9.100788821312074, 8.99665974345779, 8.981321823586548, 9.014670004112027, 9.045133217832985,
               9.064069076713762, 9.138480245200025, 9.16111754270628, 9.04933591228587, 9.04176645652599,
               9.095946531727973, 9.068640229541325, 8.822834636004972, 8.841201506864559, 8.906966776245449,
               8.865412648810345, 9.027006199869051, 9.056437729752107, 9.070877806391275, 8.883405942751354,
               8.962718180681948, 9.0734082520402, 9.035799362331563, 9.174643411860055, 9.196828405112209,
               9.173272551799638, 9.190252472706636, 9.243839758457643, 9.370333258307951, 9.398081266805022,
               9.331889793032566, 9.280522425036459, 9.210872929764376, 9.106197570970108, 9.166762925159194,
               9.276567125836856, 9.06329992784101, 9.011089811578744, 9.041914757639677, 9.239774529591678,
               9.444951617189261, 9.278916831951575, 9.337751835447754, 9.273466584434606, 9.138455281963717,
               9.109151942063267, 9.123601719516012, 9.131245565184354, 9.092973387596404, 9.112590290408274,
               9.119779947661716, 9.277914240766028, 9.316166331732028, 9.263994558829811, 9.401564011713132,
               9.429612452458926, 9.206836985301598, 8.13978779891904, 8.161394995151394, 8.180853344419386,
               8.096091165094192, 8.242140912582162]
    CO_mae = [0.0807392683760434, 0.09061447930006118, 0.11056241149381626, 0.13502152009331916, 0.12141769848030771,
              0.12739147506746282, 0.12856256278906164, 0.14794495443322522, 0.15480126926029225, 0.15295104108018778,
              0.15662486692628505, 0.15003713959320966, 0.14850378100477363, 0.1496337640943278, 0.1550314493769744,
              0.1585203477606473, 0.1698859591641374, 0.16200291240068482, 0.16249710059283357, 0.1629388915359724,
              0.15856294873093013, 0.15721547511621858, 0.15099028619136218, 0.15303728941743888, 0.15215051003943095,
              0.15629952431976873, 0.16011983652617942, 0.17053241496166452, 0.16919522555228722, 0.16497604744837108,
              0.17463644107103765, 0.17515990287259509, 0.1696732909957367, 0.17281734408599606, 0.1679084839798984,
              0.17299741008515765, 0.16506907672374146, 0.16589609208753883, 0.17029889100204887, 0.17047196395866313,
              0.15155014079167103, 0.15134895815439683, 0.1852528553947902, 0.1680488090431713, 0.1622920080346435,
              0.15072499055157373, 0.15201763177039762, 0.17538837691991882, 0.1766211792252894, 0.15955443251899762,
              0.17030719889526647, 0.17359643216379936, 0.17338344390622185, 0.16467419246895268, 0.16200212711958736,
              0.1546492991509062, 0.15842415964202386, 0.17188603024378438, 0.16079773471478992, 0.17742552774522946,
              0.16006366886721862, 0.16587315075988882, 0.15952294650417895, 0.16446708747959315, 0.16758253096079626,
              0.1551225603148003, 0.1555662781307129, 0.14841587187969676, 0.14690266528140955, 0.14814755830079288,
              0.15056576764808324, 0.1514135336640312]
    O3_mae = [7.838287223997271, 11.024490227277871, 17.04436100494881, 18.50902242021116, 15.058277223615013,
              20.226338168483057, 20.14330407810364, 20.371036186386966, 20.572585770540133, 20.913630110801677,
              21.02283023507794, 21.220610002116167, 21.247626915141268, 21.179162205665293, 21.248041661992126,
              21.180332613012084, 21.149966297428943, 21.500033278168623, 21.229228249426544, 21.0590926962828,
              21.227191264975765, 21.421683796823515, 21.38466243818718, 21.16897329277892, 21.238223148713786,
              21.09081982225065, 21.272671527971692, 21.318975107540982, 21.149436104868126, 21.511637537682788,
              21.458916991048692, 21.263395931760062, 21.2666532261372, 21.628161228461565, 21.788996346262838,
              22.049011869805465, 22.055634424414762, 22.220529271045464, 22.288968075759005, 22.094539604534027,
              22.058396380609537, 22.152544360240274, 22.06219136584887, 21.79557751006763, 21.83699031863483,
              21.819496498734527, 22.089001776721815, 22.15401785313237, 21.968265947821294, 21.853745551495823,
              21.974784103517656, 21.93455275307733, 22.06413053742953, 21.918978454231798, 22.0562230966988,
              22.081361807770325, 22.2323277322248, 22.11740679134255, 22.164871485111707, 22.25934807396308,
              22.494528036095307, 22.742408937564498, 22.514115228995525, 22.51300077946381, 22.437667904739097,
              22.12003164348047, 21.589517216600346, 21.63405387655461, 21.79610373526587, 18.374694961990745,
              18.252455504197172, 18.173167540976554]
    PM2_5_mae = [4.377950870005711, 6.363386390236066, 7.475153992730219, 10.961352089167793, 11.444674561368743,
                 12.042087419398749, 12.27195111084772, 12.363474470237103, 12.448985253178678, 12.743647514742673,
                 12.815436044180267, 12.911361712259719, 13.16226921858948, 13.27705084182115, 13.31014916306693,
                 13.471131525063985, 13.60345921286778, 13.62860665144325, 13.860895772355205, 13.817719841680653,
                 13.652665858339862, 13.740252716366246, 13.654394778613986, 13.693793181530713, 13.977716982395641,
                 14.070193191731217, 14.240927317319182, 14.40670003230395, 14.455309568583298, 14.47927862746704,
                 14.418682396639525, 14.51153608168859, 14.629606790113247, 14.662610001891707, 14.645304679451943,
                 14.530889444587608, 14.547027333651021, 14.641679023716165, 14.791271902670811, 14.883100081371579,
                 14.972627291917487, 14.918780057302262, 14.836450055758807, 14.812651052182506, 14.893807087963063,
                 14.77643538386302, 14.716945999916414, 14.669127591078432, 14.899684403106939, 14.890595661343216,
                 14.864019378401782, 14.921437711976182, 14.72659279534546, 14.802785575244945, 14.746912483498116,
                 14.707891951520729, 14.755826192022363, 14.792319125001704, 14.921752005544253, 14.790564232829698,
                 14.821407667778963, 14.742107156935903, 14.871615160443143, 14.83947213625557, 14.789278126681427,
                 14.75096102178105, 14.815145628742323, 14.821093738481672, 14.692639653928747, 11.11780603187241,
                 11.369213955108272, 11.498831850617503]
    PM10_mae = [5.638773603384198, 8.027485923210863, 9.604627475392915, 10.713123004530983, 11.595303519151184,
                16.35054375970729, 16.66035377179945, 17.0324538674133, 17.20290106688926, 17.63358307523129,
                17.980446349877937, 18.118159152583473, 18.388619310564877, 18.539491905661464, 18.80720095303812,
                19.094803224545025, 19.18083711709958, 19.231557958672255, 19.209650469548453, 19.150827563459,
                19.1123923479709, 19.0930855677051, 19.044150971191847, 19.143353063886867, 19.31930756531865,
                19.537151952709454, 19.615449611677107, 19.91124415120171, 20.226504395877427, 20.131935326065275,
                20.06896734639274, 20.111801987927173, 20.31797665540663, 20.530638304488942, 20.814997971221356,
                21.06871189843955, 21.21944174635061, 21.271729792426015, 21.14864662755533, 21.351154777988675,
                21.266043119705646, 21.269857162127835, 21.424194609886193, 21.47427065717556, 21.283822439145524,
                21.175434453764833, 20.95178118129847, 20.95519525166827, 21.280492802108, 21.178889282853895,
                21.123732977932416, 21.382791901784838, 21.590164224067557, 21.646881615231383, 21.364539420295515,
                21.427663452167028, 21.714548078013788, 21.518293812279236, 21.450231387668204, 21.457289144842097,
                21.523639163026044, 21.53242143615402, 21.653873201183956, 21.886659433578206, 21.670084813938875,
                21.559426591521216, 21.51710478671748, 21.416371120611053, 16.2830112152097, 16.28064660854357,
                16.11695250146608, 16.008983953789553]

    # 前三天
    for i in range(72):
        # 加载模型
        model = pickle.load(
            open(
                "/usr/xgboost/models/sichuan_model/pred72(+GFS)/leshan/{}/AutoML_{}_{}(h_gfs_leshan_pred72).dat".format(
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
                "/usr/xgboost/models/sichuan_model/pred72(+GFS)/leshan/{}/AutoML_{}_{}(h_gfs_leshan_pred72(4d_10d)).dat".format(
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
    data["tem"] = weather_data["TEM"][(weather_data["CITY"] == "乐山市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press'] = weather_data['PRS'][(weather_data["CITY"] == "乐山市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['hum'] = weather_data['RHU'][(weather_data["CITY"] == "乐山市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press_change'] = weather_data['PRS_CHANGE_24H'][(weather_data["CITY"] == "乐山市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_spd'] = weather_data['WIN_S_INST_MAX'][(weather_data["CITY"] == "乐山市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_dir'] = weather_data['WIN_D_INST_MAX'][(weather_data["CITY"] == "乐山市") & (weather_data["HOUR"].isin([23, 0]))].mean()
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
    pred_data = pred_data[pred_data["Area"] == "乐山市"]

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
    filename = "output_leshan"
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
                    for i in range(24-len(lastyear)):
                        lastyear.append(lastyear[-1])
                if len(lastyear) == 24:
                    hourly_data["O3_lastyear"][i: i+24] = lastyear
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



