import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import warnings
import random


warnings.filterwarnings("ignore")

start_time = time.time()

d1 = pd.date_range(start="{} 20:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d")) + timedelta(days=-1)),
                   end="{} 20:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d")) + timedelta(days=-1) + timedelta(days=+10)),
                   freq="6H")
d1 = pd.DataFrame(d1, columns=["TimePoint"])
d2 = pd.date_range(start="{} 20:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d")) + timedelta(days=-1)),
                   end="{} 00:00:00".format(pd.to_datetime(pd.datetime.now().strftime("%Y-%m-%d")) + timedelta(days=-1) + timedelta(days=+11)),
                   freq="H")
d2 = pd.DataFrame(d2, columns=["TimePoint"])

# d1 = pd.date_range(start="2022-09-28 23:00:00", end="2022-10-08 23:00:00", freq="3H")
# d1 = pd.DataFrame(d1, columns=["TimePoint"])
# d2 = pd.date_range(start="2022-09-28 23:00:00", end="2022-10-09 00:00:00", freq="H")
# d2 = pd.DataFrame(d2, columns=["TimePoint"])

data1 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_1.csv", sep=',')
data2 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_2.csv", sep=',')
data3 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_3.csv", sep=',')
data4 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_4.csv", sep=',')
data5 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_5.csv", sep=',')
data6 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_6.csv", sep=',')
data7 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_7.csv", sep=',')
data8 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_8.csv", sep=',')
data9 = pd.read_csv("/usr/xgboost/deploy/gfs_process/gfs_extract_9.csv", sep=',')
all_timepoint = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9])

all_timepoint = all_timepoint.groupby(by="StationCode")

end_time = time.time()

# print(random.sample(range(1, 10), 3))

# path_list = os.listdir("/usr/xgboost/GFS_data/{}06".format(pd.to_datetime(pd.datetime.now().strftime("%Y%m%d")) + timedelta(days=-1)))
# for path in path_list:
#     if "idx" in path:
#         os.remove("/usr/xgboost/GFS_data/{}06/{}".format(pd.to_datetime(pd.datetime.now().strftime("%Y%m%d")) + timedelta(days=-1), path))




