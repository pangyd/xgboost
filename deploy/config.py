import os
import argparse
from sqlalchemy import create_engine
from urllib.parse import quote
import json
import pymssql
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./dataset/aoti.xlsx', help='set your excel')
parser.add_argument('--station', type=str, default='aoti', help='set your station')
parser.add_argument('--pollutant', type=str, default='PM10', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant1', type=str, default='SO2', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant2', type=str, default='NO2', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant3', type=str, default='CO', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant4', type=str, default='O3', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant5', type=str, default='PM2.5', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant6', type=str, default='PM10', help='set your pollutant, i.e. SO2, NO2, CO, PM10, PM2.5, O3')
parser.add_argument('--pollutant_list', type=list, default=['SO2', 'NO2', 'CO', 'O3', 'PM2.5', 'PM10'])
parser.add_argument('--best_model_list', type=list, default=[9, 9, 6, 9, 9, 6])
parser.add_argument('--best_model', type=int, default='6', help='the label of best model')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or cpu')
parser.add_argument('--filter_window', type=int, default=21, help='set savgol filter window')    # 平滑滤波:越大平滑效果越明显，越小越接近原始曲线
parser.add_argument('--filter_rank', type=int, default=2, help='set filter rank')   # 越小平滑效果越明显，越大越接近原始曲线
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--iteration', type=int, default=1, help='each para training iteration')
parser.add_argument('--train_window', type=int, default=1, help='iterative train window, start from 1, step 3')
parser.add_argument('--prediction_window', type=int, default=1, help='output seqence')
parser.add_argument('--window_step', type=int, default=1, help='window_seq_step')
parser.add_argument('--time_budget', type=int, default=600, help='training time of AutoML')
parser.add_argument('--max_models', type=int, default=20, help='the number of random_search')
args = parser.parse_args()


def connect_database():
    # 连接数据库
    user = "sa"
    password = "sun123!@"
    host = "202.104.69.206:18710"
    database = "EnvDataChina_Beta"
    engine = create_engine("mssql+pymssql://" + user + ":" + quote(password) + "@" + host + "/" + database)

    # 获取参数表
    config = pd.read_sql("Parameter_Config", engine)
    # 把参数列转换为字典
    pol = json.loads(config["ConfigsString"][0])
    window1 = json.loads(config["ConfigsString"][1])
    window2 = json.loads(config["ConfigsString"][2])
    model_parameter = json.loads(config["ConfigsString"][3])

    return pol, window1, window2, model_parameter

