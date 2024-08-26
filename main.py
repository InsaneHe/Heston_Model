import pandas as pd
from BSM_Model import BSM_Model
from Heston_Model import Heston_Model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main():
    data = pd.read_csv("上证50ETF期权数据.csv")  # 读取原始数据

    date_start_train = 20200912  # 指定待训练数据的开始日期为2020年9月12日
    date_end_train = 20201212  # 指定待训练数据的结束日期为2020年12月12日
    option = data[
        (data["交易日期"] >= date_start_train) & (data["交易日期"] <= date_end_train)
    ]  # 选择指定日期的数据

    option = option[option["看涨看跌类型"] == "C"]  # 训责看涨期权类型
    option = option[
        [
            "交易日期",
            "执行价格",
            "剩余到期时间（年）",
            "上证50ETF价格",
            "无风险收益率（shibor)",
            "期权收盘价",
        ]
    ]
    option.columns = ["TT", "K", "t", "s0", "r", "c"]  # 将数据列标题修改为指定列标题

    # 运行BSM模型
    S = option["s0"].values
    K = option["K"].values
    T = option["t"].values
    r = option["r"].values
    sigma = 0.2

    # 把属于同一交易日期的误差率取平均值，得到BSM模型下每天的平均误差率
    bsm_value = []
    for i in range(len(S)):
        bsm = BSM_Model(S[i], K[i], T[i], r[i], sigma)
        bsm_value.append(bsm.price())

    # 计算误差率
    bsm_error = abs(bsm_value - option["c"].values) / option["c"].values

    # 将误差率添加到数据框中
    option["bsm_error"] = bsm_error

    # 按交易日期分组并计算平均误差率
    bsm_error_mean = option.groupby("TT")["bsm_error"].mean()

    # 打印BSM模型的平均误差率
    print("BSM模型的平均误差率为：", bsm_error_mean.mean())

    # Heston模型
    # 拟合出的参数，在这里输入BestPara.py中拟合出的参数
    v0 = 0.023167021173214758
    theta = 0.049339253270474184
    kappa = 1.5801765870642694
    sigma_v = 0.2129099307815552
    rho = -0.5446647060297903
    # 运行Heston模型
    # 把属于同一交易日期的误差率取平均值，得到HES模型下每天的平均误差率
    heston_value = []
    for i in range(len(S)):
        heston = Heston_Model(K[i], T[i], S[i], r[i], v0, theta, kappa, sigma_v, rho)
        heston_value.append(heston.Call_Value())

    # 计算误差率
    heston_error = abs(heston_value - option["c"].values) / option["c"].values

    # 将误差率添加到数据框中
    option["heston_error"] = heston_error

    # 按交易日期分组并计算平均误差率
    heston_error_mean = option.groupby("TT")["heston_error"].mean()

    # 打印Heston模型的平均误差率
    print("Heston模型的平均误差率为：", heston_error_mean.mean())

    # 可视化
    # 将TT列转换为日期格式
    option["TT"] = pd.to_datetime(option["TT"], format="%Y%m%d")
    # 创建一个新的数据框来存储每个模型的平均误差率
    error_df = pd.DataFrame(
        {
            "TT": bsm_error_mean.index,
            "BSM Error": bsm_error_mean.values,
            "Heston Error": heston_error_mean.values,
        }
    )
    error_df["TT"] = pd.to_datetime(error_df["TT"], format="%Y%m%d")

    # print(error_df)

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.plot(error_df["TT"], error_df["BSM Error"], label="BSM模型平均误差率")
    plt.plot(error_df["TT"], error_df["Heston Error"], label="Heston模型平均误差率")

    # 设置横坐标格式为日期，原来是20200912这种格式，转换为2020-09-12

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # 显示中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.xlabel("交易日期")
    plt.ylabel("平均误差率")
    plt.title("BSM模型与Heston模型的平均误差率随时间变化图")
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.xticks(rotation=45)  # 横坐标标签旋转45度
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()

    # 绘制柱状图，比较两个模型的平均误差率
    bsm_error_mean_value = bsm_error_mean.mean()
    heston_error_mean_value = heston_error_mean.mean()

    # 创建一个数据框来存储平均误差率
    error_comparison_df = pd.DataFrame(
        {
            "模型": ["BSM模型", "Heston模型"],
            "平均误差率": [bsm_error_mean_value, heston_error_mean_value],
        }
    )

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(
        error_comparison_df["模型"],
        error_comparison_df["平均误差率"],
        color=["blue", "green"],
    )

    # 显示中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.xlabel("模型")
    plt.ylabel("平均误差率")
    plt.title("BSM模型与Heston模型的总平均误差率比较")
    plt.ylim(
        0, max(bsm_error_mean_value, heston_error_mean_value) * 1.2
    )  # 设置y轴范围，使图表更美观
    plt.grid(axis="y")  # 仅显示y轴的网格线

    # 显示数值标签
    for i, v in enumerate(error_comparison_df["平均误差率"]):
        plt.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
