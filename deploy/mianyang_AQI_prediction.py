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
    SO2_mae = [1.5069136120185718, 1.6423939718835003, 1.8236840758615855, 1.941786294763359, 2.0266901125211354,
               2.0458165383194533, 2.080639812993858, 2.143990001977514, 2.1231353511471602, 2.134922267800928,
               2.1442460881994228, 2.16166861352283, 2.2271622894425622, 2.218359117831809, 2.214207670155887,
               2.2361602963153073, 2.2527009164867646, 2.263743987782421, 2.2722660648516517, 2.267957160970409,
               2.234767283051274, 2.2176058423221803, 2.2240486340268606, 2.2219194762230434, 2.2522402371737176,
               2.238188815982448, 2.275246067542716, 2.285407158246667, 2.3601982095700382, 2.4091732105339525,
               2.4343688034240403, 2.4289450527017307, 2.437584289481403, 2.459847017359073, 2.4535534474418257,
               2.4501633330041543, 2.456195124469958, 2.506204800335282, 2.5030717340017414, 2.5175507699301316,
               2.5019361039383208, 2.4578656936642425, 2.4275241448984795, 2.0404769229637645, 2.3979816359168744,
               2.4216055934262255, 2.421554431909745, 2.4024415278078948, 2.416409489395118, 2.3876111002632414,
               2.391560673238718, 2.426987675661489, 2.462799899085884, 2.436473980708848, 2.482875440307117,
               2.5006089929097164, 2.481148427470407, 2.482358726415453, 2.5092966540985606, 2.5735208234423306,
               2.4823733800534273, 2.5015290592998416, 2.518759736853807, 2.486187074343445, 2.4918371303119136,
               2.4446743868901213, 2.5036731109209547, 2.5099515858549566, 2.535816825680246, 2.5113155950687718,
               2.1140942149417987, 2.0768966157978155]
    NO2_mae = [7.835780380969677, 9.597594006292885, 10.468293920740681, 11.344476980549969, 11.93457272790914,
               12.50750361708542, 12.763073813010227, 12.851557487347172, 12.432766003275209, 12.482216800378277,
               12.482029739853148, 12.700351085486403, 12.787616050232698, 12.912955436924182, 12.968095062543465,
               13.03334499606885, 13.038589989686889, 13.053947352296742, 13.025051380307756, 12.984070044570126,
               12.859280357468586, 12.53815078398288, 12.309289107874418, 12.190106965683597, 12.488786676451317,
               12.615817083952962, 12.83255177588399, 12.8813579291846, 12.810099265304325, 12.624247297833666,
               12.89684777682317, 12.63736305496785, 12.766867328770745, 12.77252835637306, 12.952204968806146,
               12.80572191002065, 12.823993956210275, 12.929739498919055, 13.122336479421124, 13.107355003098641,
               13.039864341024682, 13.24853563423681, 13.247762922977152, 13.04559995411276, 13.068599633685784,
               12.81206122275161, 12.616498147418945, 12.718829175509756, 12.754831979939013, 12.719199410418275,
               12.78457262607904, 12.88848476478236, 12.953374934496713, 12.931788130926353, 12.900120528706893,
               12.737151412457262, 12.860957207964109, 12.752521823798107, 12.88288018382643, 12.962083597016106,
               13.04527951681577, 12.891939533067445, 13.115433409552796, 12.797997670275771, 12.86288557872222,
               12.87128151601338, 12.721856023439306, 12.589545827622324, 12.777113000677195, 12.837818769845342,
               12.759127252508131, 11.509099425484987]
    CO_mae = [0.08133845504320714, 0.10930404269969174, 0.12866984452427563, 0.13919663831960497, 0.14609342588548602,
              0.15175860249291298, 0.15636510754609254, 0.15564143192265656, 0.15887840541431686, 0.15942841488658396,
              0.15828150344796996, 0.1542534584513381, 0.1532771393462102, 0.15740347823989373, 0.16209652658001597,
              0.1639956554039692, 0.16343110849535408, 0.16800442306404872, 0.1683256274842006, 0.16969231521158018,
              0.16782014680850368, 0.16442501839583015, 0.1593524174826492, 0.1579285909361514, 0.16166408941962976,
              0.1657868450337107, 0.17268088148801136, 0.17228145516303928, 0.1778036266381187, 0.17623078321977945,
              0.17537169755672627, 0.1757820483084842, 0.17594230375862596, 0.17364197486642272, 0.173163026098283,
              0.17107302084168124, 0.17049537117235553, 0.17414190157779585, 0.17545981075464603, 0.1751155831220748,
              0.17554948470779552, 0.17283439797005223, 0.17359248317254972, 0.1755758293891391, 0.17516922991933076,
              0.17532857558558573, 0.17107033509787725, 0.17176553558991672, 0.17000197406273715, 0.1565159541221208,
              0.1760163928779741, 0.18028107440465146, 0.17956452138483753, 0.15762392402534856, 0.1589674186425331,
              0.18011285424943407, 0.18173555505087094, 0.18086057872132116, 0.1580003113991203, 0.17866192855813096,
              0.1774329045075974, 0.17609005030121103, 0.17631858186727437, 0.17793856484794598, 0.1575504591275289,
              0.17861622365693888, 0.1806486194120872, 0.17916085057400571, 0.160436728569917, 0.159483493838316,
              0.16011179402332296, 0.15715963624676413]
    O3_mae = [12.824365385107518, 16.104325209674226, 18.43189354663164, 19.91581875859973, 20.926666230836805,
              21.4630024414091, 21.809305655008565, 21.571428440973445, 21.43398803652558, 21.666699566958975,
              21.72870623252241, 21.845417670957907, 22.026467168118348, 22.208787113432138, 22.59492444290154,
              23.05459323793999, 23.002939721242672, 22.608440553414933, 22.198376379959296, 22.220784800288186,
              22.097989421682126, 21.80132353842311, 21.7738320832621, 21.668361904484097, 21.8246153763314,
              22.387970882112754, 22.67758234901093, 23.048666010538096, 23.062738892823326, 22.99473667290202,
              22.846950083724476, 22.428708344362374, 22.466787916821158, 22.067171580580187, 22.058796677743217,
              21.67150966464362, 21.84260386440081, 22.172329025629, 22.287179989621006, 22.67873912229286,
              22.625405763414438, 22.567515608629012, 22.497771728869587, 22.512554797520682, 22.21283522946941,
              22.211071705107464, 22.577903085518688, 22.470929822540512, 22.96099053655059, 23.076221711080528,
              23.18647664145647, 22.973464718616306, 22.763504540444163, 22.800715238311064, 22.674413742470055,
              22.602323457413018, 22.413264981808975, 22.371883323477526, 22.241968721294544, 22.324850991569882,
              22.28798806879702, 22.10339065394688, 22.517180998735302, 22.66656431785508, 22.521001646826708,
              22.275599604933962, 22.22648298586103, 22.10413916544737, 21.819786918339595, 22.051209643293703,
              22.141073106912646, 22.13124220025681]
    PM2_5_mae = [7.9068792533630505, 8.855157781439004, 9.721071541448719, 10.267935699942955, 10.998466208509369,
                 11.694987313314776, 11.915407222935249, 12.167825524595719, 12.448090095766368, 12.560268686803495,
                 12.693221223725788, 12.782560658529942, 12.877208068553228, 13.124681221261952, 13.160454261850393,
                 13.205692882421937, 13.327918102720112, 13.470616902979895, 13.641607416365215, 13.722354250537942,
                 13.710133124195384, 13.487445272828584, 13.591299608753813, 13.625569402230717, 13.729313203058672,
                 13.784895129623996, 13.85834579894743, 14.105476507533833, 13.997342818696113, 14.075232474538439,
                 14.222323190330178, 14.214707207478378, 14.329118496628437, 14.40179044566677, 14.302532849398824,
                 14.330610504830805, 14.272778922261047, 14.28956981218648, 14.430106036413079, 14.558791264797332,
                 14.463921069031443, 14.434732822871192, 14.37023370491167, 14.351534576914835, 14.434858758200944,
                 14.497563194734033, 14.602625047152777, 14.64741709103476, 14.591500199846259, 14.586784050190161,
                 14.68336893695013, 14.920319876041207, 14.90572040870048, 14.82341653228599, 14.78814308290011,
                 14.891118349440042, 14.997065944453448, 14.922985071450341, 14.953037612075308, 14.857947611090816,
                 14.554106972609496, 14.573047136534601, 14.722775416544978, 14.646270369673955, 14.538564961838599,
                 14.63045295190906, 14.697140044590542, 14.768448329428788, 14.755399608319818, 14.807459803407511,
                 14.89933868366811, 14.767981143792879]
    PM10_mae = [12.366331435653777, 13.530052264139906, 14.921904086775525, 16.156455939147474, 17.348924882915554,
                18.099211485305673, 18.556755134452914, 18.95242675623165, 19.20740801191848, 19.614898174573103,
                19.859192232186388, 20.245196359157404, 20.43274333848166, 20.834407372303147, 20.940885126775715,
                20.94334067335488, 21.051548680679094, 21.04874373556623, 20.967304552582082, 20.854091477939253,
                20.757239245755663, 20.71916620817784, 20.927866153298513, 21.2115185378253, 21.297740535542733,
                21.495661075792952, 21.63119307627789, 21.759672654637907, 21.841291212436307, 22.116064468050403,
                22.184610131178424, 22.274026502866093, 22.370450752465587, 22.450575829521114, 22.49419092791168,
                22.329629578471785, 22.568556544921545, 22.58269115113952, 22.751049520781525, 22.925698766690815,
                23.213228884108645, 23.161233662090694, 22.868247238237547, 22.812761641143044, 22.910774835986317,
                23.060059914900837, 23.080626682649232, 22.78898882725875, 22.881047780890636, 22.803655785204228,
                22.983423682224508, 23.07821517735793, 23.200119051864664, 23.194882806687563, 23.31098886120635,
                23.387530571127417, 23.418333604062795, 23.55853809931118, 23.896855138594862, 24.154374405145585,
                24.20432566474835, 24.250586987713223, 24.128646755865653, 24.331252824290978, 24.316756961244348,
                24.21593939424171, 24.174239568249376, 24.08610725468635, 23.98015302870853, 23.948075901566757,
                24.132591792375162, 19.69093167065997]

    # 前三天
    for i in range(72):
        # 加载模型
        model = pickle.load(
            open(
                "/usr/xgboost/models/sichuan_model/pred72(r2)/mianyang/{}/AutoML_{}_{}(r2_gfs_mianyang_pred72).dat".format(
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
                "/usr/xgboost/models/sichuan_model/pred72(r2)/mianyang/{}/AutoML_{}_{}(r2_gfs_miangyang_pred72(4d_10d)).dat".format(
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
                          'atmosphere_tcc', 'surface_hpbl', 'surface_sp', 'surface_t', 'surface_gust', 'surface_prate',
                          'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_850_v',
                          'isobaric_700_v', 'isobaric_500_gh', 'quarter', 'month', 'hour', 'SO2_lastday',
                          'O3_lastday', 'StationCode']]

    if pollutant == "NO2":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'O3', 'wind_spd', 'wind_dir', 'atmosphere_tcc',
                          'surface_hpbl', 'surface_sp', 'surface_t', 'surface_gust', 'middleCloudLayer_mcc',
                          'lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_850_v', 'isobaric_500_gh', 'quarter',
                          'month', 'hour', 'NO2_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "CO":
        data = pred_data[['NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'atmosphere_tcc',
                          'surface_hpbl', 'surface_t', 'surface_gust', 'isobaric_850_u', 'isobaric_850_v',
                          'isobaric_500_gh', 'quarter', 'month', 'hour', 'CO_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "O3":
        data = pred_data[['NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'tem', 'hum',
                          'TEM_MAX_24H', 'TEM_MIN_24H', 'surface_sp', 'surface_t', 'surface_gust', 'surface_hpbl',
                          'surface_SUNSD', 'surface_prate', 'isobaric_500_gh', 'isobaric_500_u', 'quarter', 'month',
                          'hour', 'vocation', 'weekday', 'O3_lastyear', 'O3_lastday', 'StationCode']]

    if pollutant == "PM2.5":
        data = pred_data[['NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'TEM_MAX_24H',
                          'TEM_MIN_24H', 'atmosphere_tcc', 'surface_t', 'surface_gust', 'surface_sp',
                          'highCloudLayer_hcc', 'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_u',
                          'isobaric_700_u', 'isobaric_700_v', 'isobaric_500_gh', 'isobaric_500_u', 'isobaric_850_v',
                          'quarter', 'month', 'hour', 'vocation', 'weekday', 'PM2.5_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "PM10":
        data = pred_data[['NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'TEM_MAX_24H',
                          'TEM_MIN_24H', 'atmosphere_tcc', 'surface_t', 'surface_gust', 'surface_sp', 'surface_prate',
                          'highCloudLayer_hcc', 'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_u',
                          'isobaric_700_u', 'isobaric_700_v', 'isobaric_500_gh', 'isobaric_500_u', 'isobaric_850_v',
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
    data["tem"] = weather_data["TEM"][(weather_data["CITY"] == "绵阳市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press'] = weather_data['PRS'][(weather_data["CITY"] == "绵阳市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['hum'] = weather_data['RHU'][(weather_data["CITY"] == "绵阳市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_spd'] = weather_data['WIN_S_INST_MAX'][(weather_data["CITY"] == "绵阳市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_dir'] = weather_data['WIN_D_INST_MAX'][(weather_data["CITY"] == "绵阳市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data = wind_spd_dir(data)
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
    pred_data = pred_data[pred_data["Area"] == "绵阳市"]

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
    filename = "output_mianyang"
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
                print(pollutant)
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



