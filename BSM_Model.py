from scipy.stats import norm
import numpy as np

# 计算公式参考：https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

# 计算BSM模型的期权价格
class BSM_Model:

    def __init__(self, S, K, T, r, sigma):
        self.S = S  # 当前股票价格
        self.K = K  # 行权价
        self.T = T  # 到期时间
        self.r = r  # 无风险利率
        self.sigma = sigma  # 波动率，该数据通常由历史数据计算得到，一般取20%左右
        self.option_type = "call"# 期权类型

    # d1: (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    # d2: d1 - sigma*sqrt(T)
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        # 看涨：value= S*N(d1) - K*e^(-rT)*N(d2)
        if self.option_type == "call":
            return self.S * norm.cdf(self.d1()) - self.K * np.exp(
                -self.r * self.T
            ) * norm.cdf(self.d2())
        # 看跌：value= K*e^(-rT)*N(-d2) - S*N(-d1)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(
                -self.d2()
            ) - self.S * norm.cdf(-self.d1())
