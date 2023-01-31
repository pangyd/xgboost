import os
import pandas as pd
import datetime
import shutil

filename = str((pd.to_datetime(pd.datetime.now()) + datetime.timedelta(days=-3)).strftime("%Y%m%d")) + "12"

shutil.rmtree("/usr/xgboost/GFS_data/{}".format(filename))
