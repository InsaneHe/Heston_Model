# 上证50ETF数据

import tushare as ts
import pandas as pd

# 设置tushare的token
ts.set_token("ed0d7318d46eb76b094b3d223602b1df610e6eea1963734dee853117")

# 初始化pro接口
pro = ts.pro_api()

# 获取期权基本信息
option_basic = pro.opt_basic(
    exchange="SSE",
    fields="ts_code,name,call_put,exercise_price,maturity_date,list_date,delist_date",
)

# 获取期权每日交易数据
option_daily = pro.opt_daily(
    exchange="SSE", fields="ts_code,trade_date,close,vol,amount"
)

# 获取上证50ETF每日数据
etf_daily = pro.fund_daily(ts_code="510050.SH", fields="trade_date,close,pct_chg")

# 获取无风险收益率（shibor）
shibor = pro.shibor(start_date="20211025", end_date="20241025", fields="date,1y")

# 检查获取的数据
print("Option Basic Data:")
print(option_basic.head())
print("Option Daily Data:")
print(option_daily.head())
print("ETF Daily Data:")
print(etf_daily.head())
print("Shibor Data:")
print(shibor.head())

# 检查日期列的格式和范围
print("Option Daily Dates:")
print(option_daily["trade_date"].unique())
print("ETF Daily Dates:")
print(etf_daily["trade_date"].unique())
print("Shibor Dates:")
print(shibor["date"].unique())

# 合并数据
data = pd.merge(option_daily, option_basic, on="ts_code")
print("After merging option_daily and option_basic:")
print(data.head())

data = pd.merge(
    data, etf_daily, left_on="trade_date", right_on="trade_date", suffixes=("", "_etf")
)
print("After merging with etf_daily:")
print(data.head())

# 确保日期格式一致
data["trade_date"] = pd.to_datetime(data["trade_date"])
shibor["date"] = pd.to_datetime(shibor["date"])

# 合并数据
data = pd.merge(
    data, shibor, left_on="trade_date", right_on="date", suffixes=("", "_shibor")
)
print("After merging with shibor:")
print(data.head())

# 将交易日期格式改为YYYYMMDD
data["trade_date"] = data["trade_date"].dt.strftime("%Y%m%d")

# 计算剩余到期时间（年）
data["maturity_date"] = pd.to_datetime(data["maturity_date"])
data["剩余到期时间（年）"] = (
    data["maturity_date"] - pd.to_datetime(data["trade_date"], format="%Y%m%d")
).dt.days / 365

# 选择需要的列并重命名
data = data[
    [
        "ts_code",
        "trade_date",
        "close",
        "vol",
        "amount",
        "name",
        "call_put",
        "exercise_price",
        "maturity_date",
        "list_date",
        "delist_date",
        "close_etf",
        "pct_chg",
        "1y",
        "剩余到期时间（年）",
    ]
]
data.columns = [
    "期权代码",
    "交易日期",
    "期权收盘价",
    "期权交易量",
    "期权交易额",
    "期权名称",
    "看涨看跌类型",
    "执行价格",
    "到期日",
    "上市日期",
    "退市日期",
    "上证50ETF价格",
    "上证50ETF日度收益率",
    "无风险收益率（shibor)",
    "剩余到期时间（年）",
]

# 数据存入csv文件
data.to_csv("上证50ETF期权数据.csv", index=False)

print("数据已成功存入csv文件")
