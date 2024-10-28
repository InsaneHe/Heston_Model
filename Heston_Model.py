import numpy as np
import scipy.integrate as spi


class Heston_Model:
    def __init__(self, K, t, S0, r, v0, theta, kappa, sigma, rho):
        self.K = K
        self.t = t
        self.S0 = S0
        self.r = r
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho

    def characteristic_function(self, phi, type):
        if type == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa

        a = self.kappa * self.theta
        x = np.log(self.S0)
        d = np.sqrt(
            (self.rho * self.sigma * phi * 1j - b) ** 2
            - self.sigma**2 * (2 * u * phi * 1j - phi**2)
        )
        g = (b - self.rho * self.sigma * phi * 1j + d) / (
            b - self.rho * self.sigma * phi * 1j - d
        )
        D = self.r * phi * 1j * self.t + (a / self.sigma**2) * (
            (b - self.rho * self.sigma * phi * 1j + d) * self.t
            - 2 * np.log((1 - g * np.exp(d * self.t)) / (1 - g))
        )
        E = (
            ((b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2)
            * (1 - np.exp(d * self.t))
            / (1 - g * np.exp(d * self.t))
        )

        return np.exp(D + E * self.v0 + 1j * phi * x)

    def integral_function(self, phi, type):
        integral = np.exp(
            -1 * 1j * phi * np.log(self.K)
        ) * self.characteristic_function(phi, type=type)
        return integral

    def P_Value(self, type):
        """计算P值（Calculate the P-value）"""
        ifun = lambda phi: np.real(self.integral_function(phi, type=type) / (1j * phi))

        # 分段积分（Integral operations）
        intervals = [(0, 10), (10, 100), (100, 1000)]
        result = 0
        for interval in intervals:
            res, err = spi.quad(ifun, interval[0], interval[1], limit=100)
            result += res

        return 0.5 + (1 / np.pi) * result

    def Call_Value(self):
        """计算看涨期权的价格（Calculating the price of a call option）"""
        P1 = self.S0 * self.P_Value(type=1)
        P2 = self.K * np.exp(-self.r * self.t) * self.P_Value(type=2)

        if np.isnan(P1 - P2):
            return 1000000
        else:
            return P1 - P2
