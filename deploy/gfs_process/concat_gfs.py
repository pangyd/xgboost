import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import warnings
import gfs_extract_1
import gfs_extract_2
import gfs_extract_3
import gfs_extract_4
import gfs_extract_5
import gfs_extract_6
import gfs_extract_7
import gfs_extract_8
import gfs_extract_9


warnings.filterwarnings("ignore")

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

data1 = gfs_extract_1.all_timepoint
data2 = gfs_extract_2.all_timepoint
data3 = gfs_extract_3.all_timepoint
data4 = gfs_extract_4.all_timepoint
data5 = gfs_extract_5.all_timepoint
data6 = gfs_extract_6.all_timepoint
data7 = gfs_extract_7.all_timepoint
data8 = gfs_extract_8.all_timepoint
data9 = gfs_extract_9.all_timepoint
all_timepoint = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10])




