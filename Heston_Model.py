# Heston_Model.py
import numpy as np
import scipy.integrate as spi


class Heston_Model:
    def __init__(self, K, t, S0, r, v0, theta, kappa, sigma, rho):
        # 由市场数据计算出的参数
        self.K = K  # 行权价
        self.t = t  # 到期时间
        self.S0 = S0  # 当前股票价格
        self.r = r  # 无风险利率

        # Heston模型参数，需要拟合
        self.v0 = v0  # 初始方差
        self.theta = theta  # 长期方差均值
        self.kappa = kappa  # 方差回归速率
        self.sigma = sigma  # 方差的波动率
        self.rho = rho  # 资产价格与方差之间的相关系数

    # 特征函数，见网址：https://en.wikipedia.org/wiki/Heston_model
    # 表达式为：exp(D + E * v0 + i * phi * x)
    def characteristic_function(self, phi, type):
        # 根据type选择不同的参数
        if type == 1:  # P1特征函数
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:  # P2特征函数
            u = -0.5
            b = self.kappa

        a = self.kappa * self.theta  # 计算a参数
        x = np.log(self.S0)  # 计算当前股票价格的对数
        # 计算d参数，d = sqrt((rho*sigma*i*phi - b)^2 - sigma^2*(2*u*i*phi - phi^2))
        d = np.sqrt(
            (self.rho * self.sigma * phi * 1j - b) ** 2
            - self.sigma**2 * (2 * u * phi * 1j - phi**2)
        )
        # 计算g参数，g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d)
        g = (b - self.rho * self.sigma * phi * 1j + d) / (
            b - self.rho * self.sigma * phi * 1j - d
        )
        # 计算D参数，D = r*phi*i*t + (a/sigma^2)*((b - rho*sigma*i*phi + d)*t - 2*ln((1 - g*exp(d*t))/(1 - g)))
        D = self.r * phi * 1j * self.t + (a / self.sigma**2) * (
            (b - self.rho * self.sigma * phi * 1j + d) * self.t
            - 2 * np.log((1 - g * np.exp(d * self.t)) / (1 - g))
        )
        # 计算E参数，E = ((b - rho*sigma*i*phi + d) / sigma^2)*(1 - exp(d*t))/(1 - g*exp(d*t))
        E = (
            ((b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2)
            * (1 - np.exp(d * self.t))
            / (1 - g * np.exp(d * self.t))
        )

        # 返回特征函数的值，表达式为：exp(D + E * v0 + i * phi * x)
        return np.exp(D + E * self.v0 + 1j * phi * x)

    # 积分部分
    def integral_function(self, phi, type):
        # 计算积分函数的值
        integral = np.exp(
            -1 * 1j * phi * np.log(self.K)
        ) * self.characteristic_function(
            phi, type=type
        )  #

        return integral

    # p值函数
    def P_Value(self, type):
        """计算P值"""
        # 定义积分函数，表达式为：exp(-i*phi*ln(K))*characteristic_function(phi, type=type)/(i*phi)
        ifun = lambda phi: np.real(self.integral_function(phi, type=type) / (1j * phi))
        # 计算积分并返回P值
        return 0.5 + (1 / np.pi) * spi.quad(ifun, 0, 1000)[0]

    def Call_Value(self):
        """计算看涨期权的价格"""
        # 计算P1和P2
        P1 = self.S0 * self.P_Value(type=1)
        P2 = self.K * np.exp(-self.r * self.t) * self.P_Value(type=2)

        # 如果结果为NaN，返回一个巨大的值表示错误，否则返回P1 - P2
        if np.isnan(P1 - P2):
            return 1000000  # 如果初始参数使定价结果为nan，就返还巨大的值，视为报错！巨大的期权价格会拉大定价误差
        else:
            return P1 - P2
