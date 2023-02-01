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
    SO2_mae = [3.4801479932340107, 1.0161034769048516, 1.0817425808552763, 1.1599203501729276, 1.194659496892862,
               1.203687972887188, 1.224200695283761, 1.2457975139746256, 1.2624516525188778, 1.2497911795461627,
               1.2579617565383605, 1.2622324000100347, 1.2882945256095342, 1.3002400812050887, 1.302953331365133,
               1.3145717537864856, 1.3204324570107968, 1.3045234512862216, 1.2794778278468413, 1.265381284387876,
               1.2609947790425784, 1.2595945400722592, 1.246768758508981, 1.2638958185147016, 1.259953387127419,
               1.2843283076788388, 1.3177286645416173, 1.2953468020991166, 1.291556259809266, 1.2925552627003654,
               1.3213111756302742, 1.3174434107939836, 1.3194443010291006, 1.3173574671916126, 1.324038845765675,
               1.32295335712249, 1.3572403060498943, 1.3644570311582571, 1.3610971052487517, 1.3732304582770987,
               1.3651030857143758, 1.3323297831855658, 1.3521964836302705, 1.3775374574213815, 1.3731451861445179,
               1.3689015223950034, 1.3793869721670544, 1.383000750514806, 1.373862763685971, 1.381393579388341,
               1.412589394866304, 1.4530761381001187, 1.4223700196033262, 1.436977528311769, 1.4393950579231012,
               1.462633848710588, 1.4479294400450142, 1.4665668914829257, 1.440291172483874, 1.4278078751252723,
               1.4237101125942617, 1.4182866186982048, 1.4274119234882847, 1.4246234411932293, 1.4077589425331332,
               1.4318185092145952, 1.4056362526918416, 1.4132430636464388, 1.403493670195444, 1.3944632300768662,
               1.1746535710951136, 1.185600033930853]
    NO2_mae = [6.501410436178101, 10.022313588442614, 11.633714918933164, 12.358504901303665, 13.019482129430585,
               13.411129935295325, 13.892638414758204, 13.695573388077475, 13.712734481876474, 14.016962028932605,
               13.908615077953737, 13.91266307137393, 13.824448688842995, 14.15064389526969, 14.07298079955152,
               14.119438725874206, 14.303727176706696, 14.1427722571392, 14.188182937868165, 14.14977747665361,
               14.137551199910966, 13.950074813330138, 13.588130513231466, 13.49032265410337, 13.753026723215472,
               14.006412008722657, 14.300189758208983, 14.596242541692906, 14.823733981415398, 14.757722284287148,
               14.697618535889815, 14.756224764946586, 14.99871193812774, 14.880971796135876, 14.937074192303692,
               14.930152817377968, 14.817037908724053, 14.585885111451246, 14.569522997907235, 14.692784652904528,
               14.967008713820045, 14.722049939625874, 14.78740985754071, 14.689733092814684, 14.757602040644134,
               14.612538450444793, 14.577403835714811, 14.440460920623757, 14.425323106217345, 14.595480540920688,
               14.705365805963217, 14.678234501706344, 14.988471020012677, 15.167293663675352, 15.164328798737836,
               15.017521696388945, 15.060808162054306, 14.944371933341209, 14.93113452291565, 14.732404202044725,
               14.647268322012154, 14.758413499325775, 14.729625980294042, 14.902425141524136, 14.681954898937288,
               14.689393547016426, 14.60801624451067, 14.54319878152768, 14.471176314037834, 14.522698712898645,
               14.139051749580291, 12.535954006919344]
    CO_mae = [0.05589428916063298, 0.09482075562511127, 0.11142428934576358, 0.12194038488500138, 0.1323950337033734,
              0.13913671883212797, 0.1461121949461445, 0.14411609437698789, 0.148867651666293, 0.1459552513771225,
              0.14604300045382299, 0.14486898785989086, 0.14567064747272204, 0.15278003893839856, 0.1649600096611232,
              0.15595484193211837, 0.1556223514017015, 0.1556019602665052, 0.1593157480686944, 0.15611438410629852,
              0.15742827321560185, 0.1551901814670335, 0.16537469105858785, 0.16568626906505554, 0.16179026282081133,
              0.17077992375400303, 0.16887571995713452, 0.1768157009041103, 0.16959846052705763, 0.17386718522272124,
              0.17246678964515721, 0.1698297307746739, 0.17000313878216222, 0.17205718150038254, 0.17240739807687688,
              0.17227415187104914, 0.17073274557789378, 0.17770432920406695, 0.1728849542901151, 0.17084067523541616,
              0.18492403286990508, 0.17144425643246283, 0.1739180602617245, 0.17027950929298377, 0.17226129508961804,
              0.17078726648239706, 0.17818830183324635, 0.17114954193985366, 0.1711811368361932, 0.17113978822815887,
              0.17387772065029572, 0.17951214727047313, 0.18735480516222958, 0.17538802030088593, 0.1723964282548418,
              0.17053182434928818, 0.16951885665994973, 0.17411676300513768, 0.1739735622403107, 0.1800000594124196,
              0.17638493488494528, 0.182427087133924, 0.18671551036681416, 0.17446513270150957, 0.174106897166268,
              0.17246484338879023, 0.17006564339990582, 0.17006959052455428, 0.16712001243458927, 0.1673941175879762,
              0.16990385072747366, 0.17848413171629798]
    O3_mae =  [7.826509304885645, 15.077473580830539, 17.47716107217299, 18.981918022576213, 19.624959281695798,
               19.921596470789556, 20.37594029854832, 20.690920351309707, 20.860602966987734, 21.313487412306664,
               21.317649893681853, 21.41277519292026, 21.758272576793118, 21.79100282627511, 22.141644258040557,
               21.853071052597283, 22.096796633066063, 22.250183765869984, 22.085562561145885, 22.078262959503252,
               21.976254532367516, 21.91168100892631, 21.694904077774343, 21.404447568315295, 21.770260594255937,
               21.704345119043975, 19.382175280515792, 21.830801902365973, 21.091522558721973, 21.407854464398504,
               21.636035946901412, 21.57647065795824, 21.803160983594772, 21.89275457902169, 21.97104250139427,
               22.181441988083776, 22.46193408782067, 22.43663605384432, 22.365106542561, 22.681270680154075,
               22.450677603680322, 22.373452027080486, 22.32570725114862, 22.233026849476666, 22.157273090742006,
               22.31724175706298, 22.254871317813805, 22.017659646041682, 21.963558864007158, 21.944346913253383,
               21.831833817760444, 21.571457276769483, 21.57452524540607, 21.977034992015884, 22.022451719043026,
               21.890656227481166, 21.876855842932496, 21.93145029772536, 21.767317751319247, 21.674757530051295,
               21.842650929341655, 21.93992151937104, 22.280233011627352, 22.243772883058416, 22.319723843274986,
               22.08519991599929, 19.115151969553168, 19.378577405044858, 19.477914282178862, 18.978168311221815,
               19.056806779474094, 19.080838182520264]
    # PM2_5_mae = [4.276617055718264, 9.178199664415471, 7.780459542684538, 8.466219706998974, 11.83218598067146,
    #              12.470271164328173, 13.0211294169748, 13.485510058513732, 13.930373380478677, 13.973800358683663,
    #              14.309352303697839, 14.713992333862954, 15.084601977029191, 15.187467286064365, 15.15656526885493,
    #              15.411146105017307, 15.434700577596024, 15.447643599769231, 15.393002834615219, 15.235698384633888,
    #              15.152204616515085, 15.161108076158506, 15.332253157551918, 15.090318384610839, 15.14682295680153,
    #              14.984794812354165, 15.138749001574803, 15.44300288893193, 15.586128633777507, 15.854047090439416,
    #              15.904062635507225, 16.030062521085053, 16.075655526154, 16.227280101060927, 15.989585236288397,
    #              16.135152896805394, 16.163922629672115, 16.289312906013883, 16.298484822957178, 16.289221840326817,
    #              16.31357669326922, 16.395638658935827, 16.551761381668552, 16.517221021992835, 16.605502961496747,
    #              16.503413445199087, 16.498285312359048, 16.408894275051626, 16.38246095496443, 16.384313548400407,
    #              16.491885843006035, 16.485904278654353, 16.636472037957667, 16.629198337028132, 16.6923179539183,
    #              16.82835042458135, 16.99717510700696, 16.81212463003753, 16.68059659461233, 16.563803763984716,
    #              16.66087491575934, 16.739051616387123, 16.708382049594967, 16.752656948085924, 16.560514779298323,
    #              16.666970642862392, 16.58190069936073, 16.577642463171745, 16.50034575076297, 13.35439493896434,
    #              13.294333894765114, 13.14182091464447]
    PM2_5_mae = [5.115875776017585, 9.405603631981542, 10.55233977274615, 11.528772725815914, 12.252968541557875,
                 12.807679033209606, 13.188111370067556, 13.485601343492183, 13.90408688387377, 14.19082170428931,
                 14.301648285596318, 14.561961418540555, 14.691226479085861, 14.718360331840685, 14.892801920419677,
                 14.948795194973055, 14.972036255438173, 15.176245910736196, 15.07008398977437, 15.044214556490001,
                 14.94864064450903, 14.932159506284062, 14.943429870662133, 15.13800523313406, 15.262700957531996,
                 15.24898110806709, 15.586575335228316, 15.766385768932611, 16.01512239118336, 16.146075133468134,
                 16.291553391871204, 16.404588634566725, 16.524303479136393, 16.474856723424615, 16.422863291921384,
                 16.317006574508593, 16.423993601041893, 16.62388841958403, 16.62762187584095, 16.625293796181708,
                 16.692357935017935, 16.594412274419494, 16.42415580109762, 16.360429533497143, 16.193539479560837,
                 16.262247232133582, 16.256561961666367, 16.386526623961277, 16.453324535116597, 16.49787269194786,
                 16.518101831923044, 16.676143283986754, 16.649892849551467, 16.676026673967783, 16.678537325614812,
                 16.690624632556432, 16.603378280467656, 16.62120270318301, 16.710311265576667, 16.70528487413419,
                 16.800948892134443, 16.848630089995627, 16.857945491213407, 17.053057690281136, 16.995556685268497,
                 16.975163748333845, 16.78303777917217, 16.809474651475337, 16.836849244615642, 16.760827279840004,
                 16.80024083331014, 16.822022447286134]
    # PM10_mae = [6.261451148449367, 9.581836391715473, 11.467572673169853, 12.981415156482853, 14.194008320327377,
    #             15.246022835097126, 19.743274098440285, 20.086211038525665, 20.4190369369904, 20.656506490217197,
    #             21.139436266638086, 21.520075483009947, 21.968852071249728, 22.001286111519807, 21.881200869815107,
    #             21.91437117799293, 21.847095895656466, 21.967283726694856, 21.82801491614945, 22.19869316358377,
    #             22.37820468418261, 22.58263549775907, 22.66411778131186, 22.670581356649866, 22.676043942835566,
    #             22.64817652716651, 23.06892153717742, 23.530647340234555, 23.644244478561273, 23.912299828331513,
    #             24.262835503807178, 24.425169249011585, 24.472053370297235, 24.46898490073344, 24.55808194322012,
    #             24.47478935726922, 24.37938432673718, 24.653721188352336, 24.77245847458589, 24.74378948935219,
    #             24.91280075864007, 25.201551417360147, 25.22085744557682, 25.180814242042477, 25.383636246351525,
    #             25.19753736849842, 25.181585923780215, 24.892679303475752, 25.029131226045397, 25.100925057761575,
    #             25.250679089995778, 25.34391695773818, 25.83731062936801, 25.678896126979904, 25.653914970772743,
    #             25.64835641561111, 25.497637016973442, 25.530744393867842, 25.735421138406785, 25.616608680935343,
    #             25.488696598361958, 25.585663618937602, 25.709916472139724, 25.49400405777617, 25.329009668659722,
    #             25.254023720519854, 25.07464870230457, 20.64731619344486, 20.850009517063484, 20.629429903402006,
    #             20.356136699792298, 20.503590787835453]
    PM10_mae = [6.254080421580535, 13.958451351129558, 15.589579261739337, 13.548088528232652, 14.486644849649167,
                18.91694135989872, 19.842043064002212, 20.325482288453053, 20.943305897821585, 21.246829266264097,
                21.408332889251326, 21.48998338569523, 21.874873254167255, 22.05621910925671, 22.115172552748245,
                22.331186003050604, 22.422140338706342, 22.427118372648092, 22.421263134902205, 22.226027846856372,
                22.17482877417833, 22.5061095807085, 22.4292453769772, 22.262936677467312, 22.35080729815519,
                22.6295892194339, 22.60332028959795, 22.945052491197604, 23.044009061683088, 23.26255837185293,
                23.599762462716317, 24.003029194299586, 24.107508381678734, 24.169640684186973, 24.207050758337108,
                24.06679991595874, 24.39509630301446, 24.52336911180215, 24.559149399301855, 24.49498237133996,
                24.696455983047617, 24.612064261468735, 24.610350209337135, 24.04675048277944, 24.257351867428923,
                24.292306986295447, 24.45021903968577, 24.25770086697075, 24.256705378132224, 24.235400638090073,
                24.2517169032736, 24.438569233049318, 24.41965522403296, 24.39807654069875, 24.5497061947194,
                24.56815535941314, 24.475870596775525, 24.455042882083866, 24.648496340350672, 24.621222550899038,
                24.472734627599177, 24.636018769506865, 24.76613012094383, 25.01377800115683, 20.197218620202175,
                20.10109493744468, 20.11294039370368, 20.239924722490564, 20.125095481542775, 20.142256222806747,
                20.012506425622444, 19.922404520203997]

    # 前三天
    for i in range(72):
        # 加载模型
        model = pickle.load(
            open(
                "/usr/xgboost/models/sichuan_model/pred72(r2)/chengdu/{}/AutoML_{}_{}(r2_gfs_chengdu_pred72).dat".format(
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
                "/usr/xgboost/models/sichuan_model/pred72(r2)/chengdu/{}/AutoML_{}_{}(r2_gfs_chengdu_pred72(4d_10d)).dat".format(
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
        data = pred_data[['SO2', 'NO2', 'CO', 'wind_spd', 'wind_dir', 'press', 'hum', 'atmosphere_tcc',
                          'surface_gust', 'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_v',
                          'isobaric_700_u', 'isobaric_700_v', 'quarter', 'month', 'hour', 'SO2_lastday', 'StationCode']]

    if pollutant == "NO2":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'atmosphere_tcc',
                          'surface_hpbl', 'surface_sp', 'surface_t', 'surface_gust', 'middleCloudLayer_mcc',
                          'lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_850_v', 'isobaric_700_u', 'isobaric_700_v',
                          'isobaric_500_gh', 'quarter', 'month', 'hour', 'NO2_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "CO":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'hum',
                          'atmosphere_tcc', 'surface_hpbl', 'surface_t', 'surface_gust', 'highCloudLayer_hcc',
                          'middleCloudLayer_mcc','lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_850_v',
                          'isobaric_500_gh', 'quarter', 'month', 'hour', 'CO_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "O3":
        data = pred_data[['NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'tem', 'hum',
                          'surface_hpbl', 'surface_sp', 'surface_t', 'surface_SUNSD', 'surface_gust', 'highCloudLayer_hcc',
                          'lowCloudLayer_lcc', 'isobaric_500_gh', 'quarter', 'month', 'hour', 'O3_lastyear', 'O3_lastday',
                          'StationCode', 'TEM_MAX_24H', 'TEM_MIN_24H', 'vocation', 'weekday']]

    if pollutant == "PM2.5":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3', 'press', 'tem',
                          'TEM_MAX_24H', 'TEM_MIN_24H', 'atmosphere_tcc', 'surface_sp', 'surface_t', 'surface_gust',
                          'highCloudLayer_hcc', 'middleCloudLayer_mcc', 'isobaric_850_u', 'isobaric_700_u', 'isobaric_700_v',
                          'isobaric_500_gh', 'isobaric_500_u', 'isobaric_500_v', 'quarter', 'month', 'hour', 'vocation', 'weekday',
                          'PM2.5_lastday', 'O3_lastday', 'StationCode']]

    if pollutant == "PM10":
        data = pred_data[['SO2', 'NO2', 'CO', 'PM2.5', 'PM10', 'wind_spd', 'wind_dir', 'O3',
                          'TEM_MAX_24H', 'TEM_MIN_24H', 'atmosphere_tcc', 'surface_sp', 'surface_t', 'surface_gust',
                          'highCloudLayer_hcc', 'middleCloudLayer_mcc', 'lowCloudLayer_lcc', 'isobaric_850_u', 'isobaric_700_u', 'isobaric_700_v',
                          'isobaric_500_gh', 'isobaric_500_u', 'isobaric_850_v', 'hour', 'vocation', 'weekday',
                          'PM10_lastday', 'O3_lastday', 'StationCode']]

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
    data["tem"] = weather_data["TEM"][(weather_data["CITY"] == "成都市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['press'] = weather_data['PRS'][(weather_data["CITY"] == "成都市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['hum'] = weather_data['RHU'][(weather_data["CITY"] == "成都市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_spd'] = weather_data['WIN_S_INST_MAX'][(weather_data["CITY"] == "成都市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data['wind_dir'] = weather_data['WIN_D_INST_MAX'][(weather_data["CITY"] == "成都市") & (weather_data["HOUR"].isin([23, 0]))].mean()
    data = wind_spd_dir(data)
    return data


def get_lastyear_data():
    now_data = pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d"))
    d_list = []
    for i in range(11):
        # print("/mnt/real/WEATHER/WEATHER_{}.csv".format(
        #     (pd.to_datetime(pd.datetime.now()) + timedelta(days=-1)).strftime("%Y-%m-%d")))
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

    # 只取成都市的数据
    pred_data = pred_data[pred_data["Area"] == "成都市"]

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
    filename = "output_chengdu"
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

    timepoint, data, la = get_data()   # （10天时间列，拿来预测的数据，前一天的数据）
    gfs_data = concat_gfs.all_timepoint   # GFS(已经分组)
    for (code, group), (code1, group1) in zip(data, la):
        try:
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
                daily_pred["TimePoint"] = [pd.to_datetime(pd.datetime.now().strftime("%Y%m%d")) + timedelta(days=+i) for
                                           i in range(10)]
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
                hourly_predict_pol_city["UniqueCode"] = list(citys["城市编码"][citys["bool"] == True]) * len(
                    hourly_predict_pol_city)
                daily_pred_city["UniqueCode"] = list(citys["城市编码"][citys["bool"] == True]) * len(daily_pred)

                hourly_predict_pol_city.to_csv(
                    "/mnt/output/AQMS/city/AQMS_hourly_{}.txt".format(pd.datetime.now().strftime("%Y%m%d")),
                    sep=",", mode="a", header=0, index=0)
                daily_pred_city.to_csv(
                    "/mnt/output/AQMS/city/AQMS_daily_{}.txt".format(pd.datetime.now().strftime("%Y%m%d")),
                    sep=",", mode="a", header=0, index=0)

                t2 = time.time()
                print("单个城市的预测时间：{}s".format(t2-t1))
        except:
            continue

    end_time = time.time()
    logging.info(f"所花费的时间：{end_time - start_time}s")
    print("花费时间：{}s".format(end_time - start_time))



