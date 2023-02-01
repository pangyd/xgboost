import matplotlib.pyplot as plt
from data_preprocess import *
from config import *
from sklearn.model_selection import train_test_split
import pickle
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
import time
import logging
import matplotlib as mpl
from all_city_station import *


# 连接数据库
pol, window1, window2, model_parameter = connect_database()

# 在服务器上加载图片
mpl.use("Agg")

warnings.filterwarnings("ignore")

# 将graphviz加入到环境变量中
os.environ["PATH"] += os.pathsep + 'E:/Graphviz/bin/'


def mean_abs_re(y_true, y_pred):
    return np.sum(abs((y_true - y_pred) / y_true)) / len(y_true)


def train_model(data, city):
    """模型训练"""
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:],
                                                        data.iloc[:, 0],
                                                        test_size=0.1)

    for i in range(window1['prediction_window']):

        automl = AutoML()

        settings = {
            "time_budget": model_parameter['time_budget'],
            "metric": model_parameter['metric'],
            "estimator_list": model_parameter["estimator_list"],
            "eval_method": model_parameter['eval_method'],
            "task": "regression",
            "log_file_name": "/suncere/pyd/PycharmProjects/output/all_city/"
                             "hourly_flaml_automl_{}_{}.log".format(city, pol['pollutant1']),
            "seed": model_parameter['seed']
        }
        automl.fit(X_train=x_train, y_train=y_train, **settings)   # y_train只能单列

        # 保存模型
        with open(
                "/suncere/pyd/PycharmProjects/models/{}/"
                "hourly_Xgboost_AutoML_{}.dat".format(city, pol['pollutant1']), "wb") as file:
            pickle.dump(automl, file)

        logging.info(f"{pol['pollutant1']}({i})的最佳参数为：{automl.best_config}")   # 最好的配置
        logging.info(f"{pol['pollutant1']}({i})的最佳损失为：{automl.best_loss}")   # 最佳的损失
        logging.info(f"{pol['pollutant1']}({i})的最好结果为：{automl.best_result}")   # 训练最佳配置所花费的时间
        logging.info(f"最佳模型为：{automl.model.estimator}")
        print("{}({})最佳参数：{}".format(pol['pollutant1'], i, automl.best_config))
        print("{}({})最佳损失：{}".format(pol['pollutant1'], i, 1 - automl.best_loss))
        print("{}({})最好结果：{}".format(pol['pollutant1'], i, automl.best_result))
        print(automl.model.estimator)

        # 特征重要性
        logging.info(f"{pol['pollutant1']}的特征重要性：{automl.model.estimator.feature_importances_}")
        # plt.title("{}的特征重要性：".format(pol['pollutant1']))
        # plt.barh(x_train.columns, automl.model.estimator.feature_importances_)
        # plt.grid()
        # plt.show()
        # plt.savefig("_{}的特征重要性.png".format(pol['pollutant1']))
        # plt.close("_{}的特征重要性.png".format(pol['pollutant1']))

        # 测试模型
        mse, mae, mar, r2 = test_model(x_test, y_test, i, city)

    return mse, mae, mar, r2


def test_model(x_test, y_test, i, city):
    """模型测试"""
    # 将测试集索引换成0：len(x_test)
    x_test = pd.DataFrame(x_test.values, columns=x_test.columns)
    y_test_col = pd.DataFrame(y_test.values)

    # 加载模型
    model = pickle.load(open("/suncere/pyd/PycharmProjects/models/{}/"
                             "hourly_Xgboost_AutoML_{}.dat".format(city, pol['pollutant1']), "rb"))

    # 模型测试
    y_pred = model.predict(x_test)
    y_pred_col = pd.DataFrame(y_pred).astype(dtype="float64")

    # 误差计算
    r2 = 1 - sklearn_metric_loss_score("r2", y_test_col, y_pred_col)
    mse = sklearn_metric_loss_score("mse", y_test_col, y_pred_col)
    mae = sklearn_metric_loss_score("mae", y_test_col, y_pred_col)
    mar = mean_abs_re(y_test_col.values, y_pred_col.values)
    logging.info(f"{pol['pollutant1']}({i})的MSE:{mse}")
    logging.info(f"{pol['pollutant1']}({i})的MAE:{mae}")
    logging.info(f"{pol['pollutant1']}({i})的R2:{r2}")
    logging.info(f"{pol['pollutant1']}({i})的MAR:{mar}")
    print("{}({})的R2={}".format(pol['pollutant1'], i, r2))
    print("{}({})的mse={}".format(pol['pollutant1'], i, mse))
    print("{}({})的mae={}".format(pol['pollutant1'], i, mae))
    print("{}({})的mar={}".format(pol['pollutant1'], i, mar))

    logging.info("-"*50)

    return mse, mae, mar, r2






