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
    SO2_mae = [1.8983042957644056, 2.542920182977908, 2.968128891750507, 3.2424235733902766, 3.334617222452337,
               3.3594609258984836, 3.6341756126749845, 3.6995770671426724, 3.673751424196784, 3.7234529696480863,
               3.6897432781553166, 3.693331700990875, 3.656385136045012, 3.601963141778246, 3.6644358192731628,
               3.594340509122636, 3.6891974911862837, 3.6974613416187867, 3.7095490046848316, 3.6784976877773676,
               3.6533794320901434, 3.6714295337266725, 3.6711407976676336, 3.7516290060686224, 3.727783234020086,
               3.761048030060501, 3.7492810459649313, 3.822562308900902, 3.7602265009135563, 3.8284741772824296,
               3.8646170891861074, 3.830221999238289, 3.7724432567282893, 3.8184000923416406, 3.768315783217002,
               3.771621571856245, 3.707001932527034, 3.691268973634855, 3.72519509514746, 3.7523411195723138,
               3.8293454199227948, 3.8753062971301406, 3.8741036886743574, 3.9258405215500525, 3.9584716108303306,
               3.9395342610779003, 3.861552776884607, 3.851021452910404, 3.8974246787843647, 3.8596458436667773,
               3.940398372885012, 3.9547853278228926, 3.8853225888075746, 3.873289283734602, 3.9314264194957995,
               3.9095512004658124, 3.9089469183186307, 3.898780408234818, 3.917295442013688, 3.9915648487591686,
               3.923624884669045, 3.9156240577041723, 3.978293974769105, 3.9441921864473457, 4.007684135986896,
               3.97386228550809, 3.748670364358878, 3.6770156423017415, 3.637025829161084, 3.6654474895576596,
               3.6875795707952967, 3.684744737710462]
    NO2_mae = [3.146419363363546, 4.440253679595941, 5.179801956910703, 7.099072534236708, 7.282036627182829,
               6.111513909018337, 7.577845111631076, 7.432296691700545, 7.496619174278349, 7.422289313721465,
               7.460695941781013, 6.660316115569248, 7.5118189769902415, 7.623059092877486, 7.668234363261455,
               7.724693836262598, 7.60117006503483, 7.4444325787159835, 7.412230551129978, 7.569106520117781,
               7.607297109193283, 7.498500392495633, 7.449055539544605, 7.380707014460505, 7.51546757495287,
               7.758905247485736, 7.774175781071742, 7.848362864750642, 8.068511886172738, 7.931217878803738,
               7.909451808440875, 8.005026981737876, 8.009540715391553, 7.982552903308752, 7.923737010771316,
               7.701775369148743, 7.708326159655872, 7.668335697065987, 7.751235969082372, 7.8190849906816275,
               7.767220378943419, 7.62279542320142, 7.495655141597606, 7.537626935045003, 7.6019884233968,
               7.483434882332597, 7.527499216593204, 6.823537531687231, 7.573688739505934, 7.69221314396369,
               7.718230301238414, 7.791331351907216, 7.88327685634705, 7.86733382284935, 7.900867782392556,
               7.8060004443062105, 7.885811150781855, 7.78063200690887, 7.792785609135653, 7.65191288031878,
               7.676943882505237, 7.749550394461519, 6.890410303374779, 6.917168681419366, 6.762916930691226,
               6.728093955017574, 6.702727791946713, 6.626357251035996, 6.757384797864624, 6.760227181027632,
               6.585354665306236, 6.578024558699823]
    CO_mae = [0.04084342393094415, 0.05853071653090777, 0.08379111227228785, 0.09147773935750304, 0.09771203533436731,
              0.08542424225092674, 0.09061535711034477, 0.10550856355038299, 0.10695193268970477, 0.10669565949754016,
              0.10722319157797987, 0.10789144108001049, 0.10950291307043387, 0.11067787533150324, 0.11441505453379705,
              0.11526235380081475, 0.11577084194226958, 0.11655865746598776, 0.11780437851181966, 0.11683790673152578,
              0.11774254536507768, 0.11744301307399958, 0.11537527200221272, 0.113514927595006, 0.115212769933197,
              0.11782110003057568, 0.12020736825947226, 0.12056954217151394, 0.12190353625846266, 0.12277628770778096,
              0.12439605255120217, 0.1251888404673227, 0.12680568355879585, 0.12701291139671664, 0.1242971191614612,
              0.12485135789848541, 0.12589451775816285, 0.1128148514823862, 0.12904913776601895, 0.12984764236153756,
              0.12790524603076162, 0.128293197971035, 0.1292799870609827, 0.13101986954554404, 0.128960878812787,
              0.12882255262506248, 0.11390409274987562, 0.11180232485585023, 0.12695849740478013, 0.1281657427541925,
              0.12686776727987195, 0.13077770590318571, 0.13259650935339046, 0.13185182987849378, 0.13338581963054527,
              0.13217722194577333, 0.1333751778322234, 0.13265522071216737, 0.13286844778102802, 0.1320488419808257,
              0.1314494984990749, 0.13125342444245647, 0.1330870412038775, 0.13108519066032293, 0.13109281055903454,
              0.11265836214531325, 0.11258613838806654, 0.11132181751392727, 0.11023862887571217, 0.11012886036671281,
              0.10965829962044912, 0.10914259180213493]
    O3_mae = [6.329158114505846, 13.12452030124346, 10.95493023472567, 16.289632858480864, 12.927042074179614,
              17.907357464217956, 14.671951841542938, 18.770048140419934, 18.747188825919597, 18.458430012741587,
              18.77930062873472, 18.97648927285837, 19.366302687809913, 19.5372596427305, 19.42839579859836,
              19.386175834295003, 19.524429032122143, 19.25875391979675, 19.240449517638115, 19.276317846311674,
              19.120013695621218, 19.084687760910715, 18.988107087468272, 18.694113163057768, 19.032863534700887,
              19.32670371984876, 19.3924493414498, 19.418076632064786, 19.61351847011772, 19.671401692609013,
              19.709612824523482, 19.745451796160538, 19.842365758160543, 19.676023130999564, 19.63473835155362,
              19.886339916604694, 19.79570538146772, 19.799602865059352, 19.998085919275734, 20.195813258873255,
              20.223558995439838, 20.100949557158337, 19.960666996466273, 19.87787692122806, 20.09193910043334,
              20.181486870329405, 20.476770469886947, 20.041698980011784, 20.049825813053822, 20.038974122349604,
              20.154031555093237, 20.125784896224328, 19.951381540509914, 20.064279104614258, 20.199850893122097,
              20.238500346451733, 20.30998625070496, 20.31255883124858, 20.229182531459188, 20.06402871377643,
              20.114982263888646, 20.20165366473967, 20.248161174755936, 20.433068288259648, 20.304398787267427,
              20.023277907905467, 20.118540294556936, 20.066071931190553, 16.87095594505518, 17.006333324741338,
              16.84415562236789, 16.882815943777672]
    PM2_5_mae = [7.5447364080027395, 8.501301490141973, 9.379187958033144, 10.04373687520173, 10.377365477097397,
                 10.73993572860052, 11.104699620684533, 11.395227596170427, 11.544525772364889, 11.678010664616892,
                 11.743857249211537, 11.95317204521934, 11.96124327282367, 12.176256425119401, 12.248705463521802,
                 12.481940653404695, 12.576647796004504, 12.60761124483513, 12.683949822415608, 12.599534627678166,
                 12.596601197122029, 12.544899696848045, 12.307545152029556, 12.292938214409915, 12.480982294415783,
                 12.553430586424474, 12.722978742823434, 12.76570587428464, 12.907207538234449, 13.013410864435308,
                 13.00706666383076, 12.885263247115898, 13.008252202499966, 12.843418804447355, 12.763705523510641,
                 12.69223938534318, 12.85272959324028, 12.987808673003787, 13.10810920916872, 13.215844653163176,
                 13.325783468455452, 13.423968308345065, 13.685267325881785, 13.498662398991446, 13.462015218622028,
                 13.42440604950061, 13.434374098789908, 13.367310721981854, 13.31497113876434, 13.455102511651138,
                 13.444317020774392, 13.263789130700257, 13.412587747172736, 13.540666944299046, 13.491739549518282,
                 13.434828332534632, 13.544731135264598, 13.589298633801103, 13.649550331067356, 13.570550807832802,
                 13.560396083625925, 13.566778970697314, 13.746987448925774, 13.714134562978893, 13.837601938160827,
                 13.655669316497777, 13.66053989571694, 13.515852576905694, 13.568508833741284, 13.458822098974524,
                 13.497053896301564, 13.590861112347689]
    PM10_mae = [7.027379227304191, 11.323685441637123, 9.941612417269589, 10.647067424385735, 11.174539867998774,
                14.422859475670945, 14.64327738512929, 14.802618412284765, 15.040269969717814, 15.068317711169836,
                15.525247199513599, 15.660699393659783, 15.912902868692932, 16.17190931157606, 16.349674834899606,
                16.291968818047867, 16.46801466844805, 16.502590734935303, 16.68009546600831, 16.851164213566456,
                16.93339466256368, 16.619555299167104, 16.33347972367735, 16.701942199143076, 16.755096734177805,
                16.921130994542715, 17.00846653550351, 17.371352027902596, 17.582597137361645, 17.55766608309379,
                17.521811140569202, 17.41394273206547, 17.44160931483284, 17.661085785878228, 17.884011382626507,
                17.990780146290973, 18.41758748560336, 18.30302528967171, 18.62464682561832, 18.30702021903461,
                18.14834090017035, 18.234385796621996, 18.43821763148491, 18.317090776336354, 18.16324505178011,
                18.179914480304863, 18.1789518650334, 18.247777885980597, 18.050272050091568, 18.316364499498103,
                18.454478196917748, 18.594483977061174, 18.448029164754477, 18.522785128238624, 18.480638685726525,
                18.252558391424383, 17.94203072980817, 18.216207006011675, 18.230769532628017, 18.198725881532663,
                18.17450048210226, 18.277959580325902, 18.39862801361586, 18.621869001983026, 18.651431890740383,
                18.656951299476876, 18.577762801765868, 18.848875469723183, 14.937155021591275, 14.921516287119562,
                15.33243810451395, 15.233787218325936]

    # 前三天
    for i in range(72):
        # 加载模型
        model = pickle.load(
            open(
                "/usr/xgboost/models/sichuan_model/pred72(+GFS)/luzhou/{}/AutoML_{}_{}(h_gfs_luzhou_pred72).dat".format(
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
                "/usr/xgboost/models/sichuan_model/pred72(+GFS)/luzhou/{}/AutoML_{}_{}(h_gfs_luzhou_pred72(4d_10d)).dat".format(
                    pollutant,
                    pollutant,
                    i - 71),
                "rb"))
        if i > 24:
            pred_data["{}_lastday".format(pollutant)][i + 1] = pred_data["{}".format(pollutant)][i - 24 + 1]
            hourly_data["{}_lastday".format(pollutant)][i + 1] = hourly_data["{}".format(pollutant)][i - 24 + 1]

        # 预测每一行的值，赋值给下一列
        x = pred_data[i:i + 2]

        # 预测
        y_pred = model.predict(x)  # 单行不能预测
        pred_data["{}".format(pollutant)][i + 2] = y_pred[-1]
        hourly_data["{}".format(pollutant)][i + 2] = y_pred[-1]


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
    data["tem"] = weather_data["TEM"][(weather_data["CITY"] == "泸州市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press'] = weather_data['PRS'][(weather_data["CITY"] == "泸州市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['hum'] = weather_data['RHU'][(weather_data["CITY"] == "泸州市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_spd'] = weather_data['WIN_S_INST_MAX'][(weather_data["CITY"] == "泸州市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_dir'] = weather_data['WIN_D_INST_MAX'][(weather_data["CITY"] == "泸州市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data = wind_spd_dir(data)
    data['press_change'] = weather_data['PRS_CHANGE_24H'][(weather_data["CITY"] == "泸州市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    return data


def get_lastyear_data():
    now_data = pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d"))
    d_list = []
    for i in range(11):
        try:
            d = pd.read_csv("/mnt/real/AIR/Station_{}-{}-{}.tsv".format(
                now_data.year - 1, now_data.month, str((now_data + timedelta(days=i-1)).day).rjust(2, "0")), sep="\t")
        except:
            d = pd.read_csv("/mnt/real/AIR/Station_{}-{}-{}.csv".format(
                now_data.year - 1, now_data.month, str((now_data + timedelta(days=i-1)).day).rjust(2, "0")), sep="\t")
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
    pred_data = pred_data[pred_data["Area"] == "泸州市"]

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
    filename = "output_luzhou"
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



