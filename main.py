import pandas as pd
from BSM_Model import BSM_Model
from Heston_Model import Heston_Model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def main():
    data = pd.read_csv("上证50ETF期权数据.csv")  # 读取原始数据（Read raw data）

    date_start_train = 20161231  # 指定待训练数据的开始日期（Specify the start date of the data to be trained）
    date_end_train = 20191231  # 指定待训练数据的结束日期（Specify the end date of the data to be trained）
    date_start_test = 20191231  # 指定验证数据的开始日期（Specify the start date of validation data）
    date_end_test = 20201231  # 指定验证数据的结束日期（Specify the end date of validation data）

    # 选择训练数据（Select training data）
    train_data = data[
        (data["交易日期"] >= date_start_train) & (data["交易日期"] <= date_end_train)# 交易日期Transaction Date
    ]
    train_data = train_data[train_data["看涨看跌类型"] == "C"]# 看涨看跌类型Type (call or put option)
    train_data = train_data[
        [
            "交易日期",
            "执行价格",
            "剩余到期时间（年）",
            "上证50ETF价格",
            "无风险收益率（shibor)",
            "期权收盘价",
        ]# “Trade Date”, ‘Strike Price’, ‘Remaining Maturity (Years)’, ‘SSE 50 ETF Price’, ‘Risk Free Yield (shibor)’, ”Option Closing Price”
    ]
    train_data.columns = ["TT", "K", "t", "s0", "r", "c"]

    # 选择验证数据（Select Validation Data）
    test_data = data[
        (data["交易日期"] >= date_start_test) & (data["交易日期"] <= date_end_test)
    ]
    test_data = test_data[test_data["看涨看跌类型"] == "C"]
    test_data = test_data[
        [
            "交易日期",
            "执行价格",
            "剩余到期时间（年）",
            "上证50ETF价格",
            "无风险收益率（shibor)",
            "期权收盘价",
        ]
    ]
    test_data.columns = ["TT", "K", "t", "s0", "r", "c"]

    # 运行BSM模型（Running the BSM model）
    S = test_data["s0"].values
    K = test_data["K"].values
    T = test_data["t"].values
    r = test_data["r"].values
    sigma = 0.2

    bsm_value = []
    for i in range(len(S)):
        bsm = BSM_Model(S[i], K[i], T[i], r[i], sigma)
        bsm_value.append(bsm.price())

    bsm_value = np.array(bsm_value)
    bsm_error = np.abs(bsm_value - test_data["c"].values) / test_data["c"].values
    test_data["bsm_error"] = bsm_error
    bsm_error_mean = test_data.groupby("TT")["bsm_error"].mean()
    print("BSM模型的平均误差率为：", bsm_error_mean.mean())# The average error rate of the BSM model is:

    # 运行Heston模型（Running the Heston model）
    v0 = 0.023167021173214758
    theta = 0.049339253270474184
    kappa = 1.5801765870642694
    sigma_v = 0.2129099307815552
    rho = -0.5446647060297903

    heston_value = []
    for i in range(len(S)):
        heston = Heston_Model(K[i], T[i], S[i], r[i], v0, theta, kappa, sigma_v, rho)
        heston_value.append(heston.Call_Value())

    heston_value = np.array(heston_value)
    heston_error = np.abs(heston_value - test_data["c"].values) / test_data["c"].values
    test_data["heston_error"] = heston_error
    heston_error_mean = test_data.groupby("TT")["heston_error"].mean()
    print("Heston模型的平均误差率为：", heston_error_mean.mean())# The average error rate of the Heston model is:

    # 可视化（visualization）
    test_data["TT"] = pd.to_datetime(test_data["TT"], format="%Y%m%d")
    error_df = pd.DataFrame(
        {
            "TT": bsm_error_mean.index,
            "BSM Error": bsm_error_mean.values,
            "Heston Error": heston_error_mean.values,
        }
    )
    error_df["TT"] = pd.to_datetime(error_df["TT"], format="%Y%m%d")

    plt.figure(figsize=(12, 6))
    plt.plot(error_df["TT"], error_df["BSM Error"], label="BSM模型平均误差率Average error rate of the BSM model")
    plt.plot(error_df["TT"], error_df["Heston Error"], label="Heston模型平均误差率Average error rate of the Heston model")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.xlabel("交易日期Transaction date")
    plt.ylabel("平均误差率Average error rate")
    plt.title("BSM模型与Heston模型的平均误差率随时间变化图Plot of average error rate over time for BSM model vs. Heston model")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    bsm_error_mean_value = bsm_error_mean.mean()
    heston_error_mean_value = heston_error_mean.mean()
    error_comparison_df = pd.DataFrame(
        {
            "模型": ["BSM模型", "Heston模型"],
            "平均误差率": [bsm_error_mean_value, heston_error_mean_value],
        }
    )

    plt.figure(figsize=(8, 6))
    plt.bar(
        error_comparison_df["模型"],# Model
        error_comparison_df["平均误差率"],# Average error rate
        color=["blue", "green"],
    )
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.xlabel("模型")
    plt.ylabel("平均误差率")
    plt.title("BSM模型与Heston模型的总平均误差率比较Comparison of Total Average Error Rate between BSM and Heston Models")
    plt.ylim(0, max(bsm_error_mean_value, heston_error_mean_value) * 1.2)
    plt.grid(axis="y")
    for i, v in enumerate(error_comparison_df["平均误差率"]):
        plt.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
