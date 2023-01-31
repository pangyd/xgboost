import xarray as xr
import cfgrib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from multiprocessing import Pool
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")


start_time = time.time()

# d1 = pd.date_range(start="{} 23:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d"))),
#                    end="{} 23:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d")) + timedelta(days=+10)),
#                    freq="3H")
# d1 = pd.DataFrame(d1, columns=["TimePoint"])
# d2 = pd.date_range(start="{} 23:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d"))),
#                    end="{} 00:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d")) + timedelta(days=+11)),
#                    freq="H")
# d2 = pd.DataFrame(d2, columns=["TimePoint"])

d1 = pd.date_range(start="2022-09-28 23:00:00", end="2022-10-08 23:00:00", freq="3H")
d1 = pd.DataFrame(d1, columns=["TimePoint"])
d2 = pd.date_range(start="2022-09-28 23:00:00", end="2022-10-09 00:00:00", freq="H")
d2 = pd.DataFrame(d2, columns=["TimePoint"])

filepath = "/usr/xgboost/GFS_data/2022092806/"
# file = "gfs.0p25.2022050100.f000.grib2"
# files = os.listdir(filepath)

files = []
for i in np.arange(9, 250, 3):
    if i == 9:
        files.append("gfs.t06z.pgrb2.0p50.f00{}".format(i))
    if 10 < i < 100:
        files.append("gfs.t06z.pgrb2.0p50.f0{}".format(i))
    if i > 100:
        files.append("gfs.t06z.pgrb2.0p50.f{}".format(i))

station = pd.read_csv('station.csv')
area = station['Area']
station_lat = station['Latitude']
station_lon = station['Longitude']
UniqueCode = station['UniqueCode']
StationCode = station['StationCode']


def _(l, atmosphere_tcc, surface_hpbl, surface_sp, surface_SUNSD, surface_t, surface_prate, surface_gust,
      highCloudLayer_hcc, middleCloudLayer_mcc, lowCloudLayer_lcc, isobaric_850_u, isobaric_850_v, isobaric_700_u,
      isobaric_700_v, isobaric_500_gh, isobaric_500_u, isobaric_500_v, single_timepoint):
    for i in l:
        atmosphere_tcc_interp = atmosphere_tcc.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()

        surface_hpbl_interp = surface_hpbl.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                  method='linear').item()
        surface_sp_interp = surface_sp.interp(longitude=station_lon[i], latitude=station_lat[i], method='linear').item()
        surface_SUNSD_interp = surface_SUNSD.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                    method='linear').item()
        surface_t_interp = surface_t.interp(longitude=station_lon[i], latitude=station_lat[i], method='linear').item()
        surface_prate_interp = surface_prate.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                    method='linear').item()
        surface_gust_interp = surface_gust.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                  method='linear').item()

        highCloudLayer_hcc_interp = highCloudLayer_hcc.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                              method='linear').item()
        middleCloudLayer_mcc_interp = middleCloudLayer_mcc.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                                  method='linear').item()
        lowCloudLayer_lcc_interp = lowCloudLayer_lcc.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                            method='linear').item()

        isobaric_850_u_interp = isobaric_850_u.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()
        isobaric_850_v_interp = isobaric_850_v.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()

        isobaric_700_u_interp = isobaric_700_u.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()
        isobaric_700_v_interp = isobaric_700_v.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()

        isobaric_500_gh_interp = isobaric_500_gh.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                        method='linear').item()
        isobaric_500_u_interp = isobaric_500_u.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()
        isobaric_500_v_interp = isobaric_500_v.interp(longitude=station_lon[i], latitude=station_lat[i],
                                                      method='linear').item()

        # 单个站点加入DataFrame
        single_timepoint = single_timepoint.append(
            {"StationCode": StationCode[i], 'UniqueCode': UniqueCode[i], 'Area': area[i],
             'atmosphere_tcc': atmosphere_tcc_interp, 'surface_hpbl': surface_hpbl_interp,
             'surface_sp': surface_sp_interp, 'surface_SUNSD': surface_SUNSD_interp,
             'surface_t': surface_t_interp, 'surface_prate': surface_prate_interp,
             'surface_gust': surface_gust_interp, 'highCloudLayer_hcc': highCloudLayer_hcc_interp,
             'middleCloudLayer_mcc': middleCloudLayer_mcc_interp, 'lowCloudLayer_lcc': lowCloudLayer_lcc_interp,
             'isobaric_850_u': isobaric_850_u_interp, 'isobaric_850_v': isobaric_850_v_interp,
             'isobaric_700_u': isobaric_700_u_interp, 'isobaric_700_v': isobaric_700_v_interp,
             'isobaric_500_gh': isobaric_500_gh_interp, 'isobaric_500_u': isobaric_500_u_interp,
             'isobaric_500_v': isobaric_500_v_interp}, ignore_index=True)
    return single_timepoint


def f(files):
    single_timepoint = pd.DataFrame(columns=['StationCode', 'UniqueCode', 'Area', 'atmosphere_tcc', 'surface_hpbl', 'surface_sp',
                                             'surface_SUNSD', 'surface_t', 'surface_prate', 'surface_gust',
                                             'highCloudLayer_hcc', 'middleCloudLayer_mcc', 'lowCloudLayer_lcc',
                                             'isobaric_850_u', 'isobaric_850_v', 'isobaric_700_u', 'isobaric_700_v',
                                             'isobaric_500_gh', 'isobaric_500_u', 'isobaric_500_v'])
    all_timepoint = pd.DataFrame(columns=['StationCode', 'UniqueCode', 'Area', 'atmosphere_tcc', 'surface_hpbl', 'surface_sp',
                                          'surface_SUNSD', 'surface_t', 'surface_prate', 'surface_gust',
                                          'highCloudLayer_hcc', 'middleCloudLayer_mcc', 'lowCloudLayer_lcc',
                                          'isobaric_850_u', 'isobaric_850_v', 'isobaric_700_u', 'isobaric_700_v',
                                          'isobaric_500_gh', 'isobaric_500_u', 'isobaric_500_v'])
    # 拼接81个GFS表
    for file in files:
        da_atmosphere = xr.open_dataset(filepath + file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'atmosphere'}})

        da_surface = xr.open_dataset(filepath + file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'surface'}})

        da_highCloudLayer = xr.open_dataset(filepath + file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'highCloudLayer'}})

        da_isobaricInhPa = xr.open_dataset(filepath + file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa'}})

        da_middleCloudLayer = xr.open_dataset(filepath + file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'middleCloudLayer'}})

        da_lowCloudLayer = xr.open_dataset(filepath + file, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'lowCloudLayer'}})

        atmosphere_tcc = da_atmosphere['tcc'].loc[90:0, 70:150]
        surface_hpbl = da_surface['hpbl'].loc[90:0, 70:150]
        surface_sp = da_surface['sp'].loc[90:0, 70:150]
        surface_SUNSD = da_surface['SUNSD'].loc[90:0, 70:150]
        surface_t = da_surface['t'].loc[90:0, 70:150]
        surface_prate = da_surface['prate'].loc[90:0, 70:150]
        surface_gust = da_surface['gust'].loc[90:0, 70:150]

        highCloudLayer_hcc = da_highCloudLayer['hcc'].loc[90:0, 70:150]
        middleCloudLayer_mcc = da_middleCloudLayer['mcc'].loc[90:0, 70:150]
        lowCloudLayer_lcc = da_lowCloudLayer['lcc'].loc[90:0, 70:150]

        isobaric_850_u = da_isobaricInhPa['u'].loc['850'].loc[90:0, 70:150]
        isobaric_850_v = da_isobaricInhPa['v'].loc['850'].loc[90:0, 70:150]

        isobaric_700_u = da_isobaricInhPa['u'].loc['700'].loc[90:0, 70:150]
        isobaric_700_v = da_isobaricInhPa['v'].loc['700'].loc[90:0, 70:150]

        isobaric_500_gh = da_isobaricInhPa['gh'].loc['500'].loc[90:0, 70:150]
        isobaric_500_u = da_isobaricInhPa['u'].loc['500'].loc[90:0, 70:150]
        isobaric_500_v = da_isobaricInhPa['v'].loc['500'].loc[90:0, 70:150]

        single_timepoint = _(range(len(StationCode[:2])), atmosphere_tcc, surface_hpbl, surface_sp, surface_SUNSD,
                             surface_t, surface_prate, surface_gust, highCloudLayer_hcc, middleCloudLayer_mcc,
                             lowCloudLayer_lcc, isobaric_850_u, isobaric_850_v, isobaric_700_u, isobaric_700_v,
                             isobaric_500_gh, isobaric_500_u, isobaric_500_v, single_timepoint)

        # partial_work = partial(_, {"atmosphere_tcc":atmosphere_tcc, "surface_hpbl":surface_hpbl, "surface_sp":surface_sp,
        #                        "surface_SUNSD":surface_SUNSD, "surface_t":surface_t, "surface_prate":surface_prate,
        #                        "surface_gust":surface_gust, "highCloudLayer_hcc":highCloudLayer_hcc,
        #                        "middleCloudLayer_mcc":middleCloudLayer_mcc, "lowCloudLayer_lcc":lowCloudLayer_lcc,
        #                        "isobaric_850_u":isobaric_850_u, "isobaric_850_v":isobaric_850_v, "isobaric_700_u":isobaric_700_u,
        #                        "isobaric_700_v":isobaric_700_v, "isobaric_500_gh":isobaric_500_gh, "isobaric_500_u":isobaric_500_u,
        #                        "isobaric_500_v":isobaric_500_v, "single_timepoint":single_timepoint})

        # gfs_dict = {"atmosphere_tcc":atmosphere_tcc, "surface_hpbl":surface_hpbl, "surface_sp":surface_sp,
        #                        "surface_SUNSD":surface_SUNSD, "surface_t":surface_t, "surface_prate":surface_prate,
        #                        "surface_gust":surface_gust, "highCloudLayer_hcc":highCloudLayer_hcc,
        #                        "middleCloudLayer_mcc":middleCloudLayer_mcc, "lowCloudLayer_lcc":lowCloudLayer_lcc,
        #                        "isobaric_850_u":isobaric_850_u, "isobaric_850_v":isobaric_850_v, "isobaric_700_u":isobaric_700_u,
        #                        "isobaric_700_v":isobaric_700_v, "isobaric_500_gh":isobaric_500_gh, "isobaric_500_u":isobaric_500_u,
        #                        "isobaric_500_v":isobaric_500_v, "single_timepoint":single_timepoint}
        # a = []
        # for i in l:
        #     a.append((i, atmosphere_tcc, surface_hpbl, surface_sp, surface_SUNSD, surface_t, surface_prate, surface_gust,
        #               highCloudLayer_hcc, middleCloudLayer_mcc, lowCloudLayer_lcc, isobaric_850_u, isobaric_850_v,
        #               isobaric_700_u, isobaric_700_v, isobaric_500_gh, isobaric_500_u, isobaric_500_v, single_timepoint))
        # pool = Pool(processes=16)
        # reulst_list = pool.map(_, a)
        # pool.close()
        # pool.join()

        # 合并16个进程的sing_timepoint
        # for single in reulst_list:
        #     single_timepoint.append(single)

        all_timepoint = all_timepoint.append(single_timepoint)
    return all_timepoint


if __name__ == "__main__":

    all_timepoint = f(files)

    all_timepoint = all_timepoint.groupby(by="StationCode")
    i == 0
    for code, data in all_timepoint:
        data.index = range(len(data))
        sole_station = pd.concat([d1, data], axis=1)
        sole_station = d2.merge(sole_station, how="outer", on="TimePoint")
        sole_station = sole_station.interpolate(method="ffill")
        print(sole_station)
        i += 1
        if i == 1:
            break

    end_time = time.time()
    print("{}s".format(end_time-start_time))

    # 删除过程中生成的idx文件
    for file in files:
        if "idx" in file:
            os.remove(filepath+"/"+file)

# 输出给总代码
all_timepoint







