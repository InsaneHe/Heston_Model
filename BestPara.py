# 利用模拟退火算法寻找Heston模型最优参数
# 目标函数为：最小化市场价格和Heston模型价格的均方误差

from Heston_Model import Heston_Model
import pandas as pd
import numpy as np
import copy


# 将x增加一个随机变动量，但会把x限制在a,b之内
def random_range(x, a, b):
    random = np.random.normal(0, 0.01 * (b - a), 1)[0]
    if x + random > b:
        random = -np.abs(random)  # 如果新值超越最大值，就将增量变为负数
    elif x + random < a:
        random = np.abs(random)  # 如果新值超越最小值，就将增量变为正数
    return x + random

# 模拟退火算法
class NG:
    def __init__(self, func, x0):
        """
        x0  -列表：函数的初始值
                [v0,kappa,theta,sigma,rho]
        func    -待求解函数
        """
        self.x = x0
        self.dim_x = len(x0)  # 解的维度
        self.func = func
        self.f = func(self.x)  # 计算y值

        self.x_best = self.x  # 记录下来历史最优解，即所有循环中的最优解
        self.f_best = self.f

        self.times_stay = 0  # 连续未接受新解的次数
        self.times_stay_max = 400  # 当连续未接受新解超过这个次数，终止循环

        self.T = 100  # 初始温度：初始温度越高，前期接受新解的概率越大
        self.speed = 0.7  # 退火速度：温度下降速度越快，后期温度越低，接受新解的概率越小，从而最终稳定在某个解附近
        self.T_min = 1e-6  # 最低温度：当低于该温度时，终止全部循环
        self.xf_best_T = {}  # 记录下接受的所有新解

        # 最初若函数值变动为delta，则认为函数值变动很大，可以产生p_expec概率接受新解
        # 若在初期便产生巨大的概率接受新解，则前期寻找新解的过程将变成盲目的随机漫步毫无意义，因此利用alpha调节概率
        self.p_expec = 0.9
        self.delta_standard = 0.7
        self.alpha = self.find_alpha()  # 调节概率因子

        self.times_delta_samller = 0  # 统计新旧最优值之差绝对值连续小于某值的次数
        self.delta_min = (
            0.001  # 当新旧最优值之差绝对值连续小于此值达到某一次数时，终止该温度循环
        )
        self.times_delta_min = (
            100  # 当新旧最优值之差绝对值连续小于此值达到这个次数时，终止该温度循环
        )

        self.times_max = 500  # 当每个温度下循环超过这个次数，终止该温度循环
        self.times_cycle = 0  # 记录算法循环总次数

        self.times_p = 0  # 统计因为p值较大而接受新解的次数

        self.xf_all = {
            self.times_cycle: [self.x, self.f]
        }  # 记录下来每一次循环产生的新解和函数值
        self.xf_best_all = {
            self.times_cycle: [self.x, self.f]
        }  # 记录下来每一次循环接受的新解和函数值

    # 温度下降，产生新温度
    def T_change(self):
        self.T = self.T * self.speed
        print(
            "当前温度为{},大于最小温度{}".format(self.T, self.T_min)
        )  # 展示当前温度和最小温度

    # 将所有的x和f、循环次数存储下来
    def save_xy(self):
        self.xf_all[self.times_cycle] = [self.x, self.f]

    # 将所有的最优x,y、循环次数存储下来
    def save_best_xy(self):
        self.xf_best_all[self.times_cycle] = [self.x, self.f]

    # 当调节因子为alpha时，函数值变动值为delta产生的接受新解概率
    def __p_delta(self, alpha):
        return np.exp(-self.delta_standard / (self.T * alpha))

    # 用二分法寻找方程解
    def __find_solver(self, func, f0):
        """
        输入：
        func    -待求解方程的函数
        f0  -float,预期函数值
        输出：
        mid -float,函数=预期函数值 的解
        """
        up = 100
        down = 0.00001
        mid = (up + down) / 2
        while abs(func(mid) - f0) > 0.0001:
            if func(down) < f0 < func(mid):
                up = mid
                mid = (mid + down) / 2
            elif func(up) > f0 > func(mid):
                down = mid
                mid = (up + down) / 2
            else:
                print("error!")
                break
        return mid

    # 最初若函数值变动为delta，则认为函数值变动很大，可以产生p_expec概率接受新解
    def find_alpha(self):
        return self.__find_solver(self.__p_delta, self.p_expec)

    # 获得新的x
    def get_x_new(self):
        random = np.random.normal(0, 1, self.dim_x)  # 新的随机增量
        return self.x + random

    # 判断是否可以接受新的解
    def judge(self):
        if self.delta < 0:  # 如果函数值变动幅度小于0,则接受新解
            self.x = self.x_new
            self.f_last = self.f  # 在最优解函数值更新之前将其记录下来
            self.f = self.f_new
            self.save_best_xy()  # 记录每次循环接受的新解
            self.get_history_best_xy()  # 更新历史最优解
            self.times_stay = 0  # 由于未接受新解，将连续未接受新解的次数归零
            print(
                "由于函数值变小新接受解{}:{}".format(self.f, self.x)
            )  # 展示当前接受的新解
        else:
            p = np.exp(-self.delta / (self.T * self.alpha))  # 接受新解的概率
            p_ = np.random.random()  # 判断标准概率
            if p > p_:  # 如果概率足够大，接受新解
                self.x = self.x_new
                self.f_last = self.f  # 在接受的新解更新之前将其记录下来
                self.f = self.f_new
                self.save_best_xy()  # 记录每次循环接受的新解
                self.get_history_best_xy()  # 更新历史最优解
                print(
                    "由于概率{}大于{}，新接受解{}:{}".format(p, p_, self.f, self.x)
                )  # 展示当前接受的新解
                self.times_p += 1  # 统计因为概率而接受新解的次数
                self.times_stay = 0  # 由于未接受新解，将连续未接受新解的次数归零
            else:
                if self.time_ == 0:
                    self.f_last = self.f  # 在接受的新解更新之前将其记录下来
                self.times_stay += 1  # 连续接受新解次数加1
                print("连续未接受新解{}次".format(self.times_stay))

    # 获得历史最优解
    def get_history_best_xy(self):
        x_array = list(
            np.array(list(self.xf_best_all.values()),dtype=object)[:, 0]
        )  # 从历史所有的最优x和f中获得所有的x
        f_array = list(
            np.array(list(self.xf_best_all.values()),dtype=object)[:, 1]
        )  # 从历史所有的最优x和f中获得所有的f
        self.f_best = min(f_array)  # 从每阶段最优的f中获得最优的f
        self.x_best = x_array[f_array.index(self.f_best)]  # 利用最优f反推最优x
        return self.x_best, self.f_best


    # 统计新旧函数值之差的绝对值连续小于此值的次数
    def count_times_delta_smaller(self):
        if self.delta_best < self.delta_min:
            self.times_delta_samller += (
                1  # 如果新旧函数值之差绝对值小于某值，则次数加1，否则归零
            )
        else:
            self.times_delta_samller = 0
        print(
            "差值{}连续小于{}达到{}次".format(
                self.delta_best, self.delta_min, self.times_delta_samller
            )
        )

    # 终止循环条件
    def condition_end(self):
        if (
            self.times_delta_samller > self.times_delta_min
        ):  # 如果新旧函数值之差绝对值连续小于某值次数超过某值，终止该温度循环
            return True
        elif (
            self.times_stay > self.times_stay_max
        ):  # 当连续未接受新解超过这个次数，终止循环
            return True

    # 在某一特定温度下进行循环
    def run_T(self):

        for time_ in range(self.times_max):
            self.time_ = time_
            self.x_new = self.get_x_new()  # 获得新解
            self.f_new = self.func(self.x_new)  # 获得新的函数值
            self.save_xy()  # 将新解和函数值记录下来
            self.delta = self.f_new - self.f  # 计算函数值的变化值
            self.judge()  # 判断是否接受新解
            self.times_cycle += 1  # 统计循环次数
            self.delta_best = np.abs(
                self.f - self.f_last
            )  # 上次函数值与这次函数值的差值绝对值
            self.count_times_delta_smaller()  # 统计新旧函数值之差的绝对值连续小于此值的次数
            if self.condition_end() == True:  # 如果满足终止条件，终止该温度循环
                print(
                    "满足终止条件：接受新解后的函数值变化连续小于{}达到次数".format(
                        self.delta_min
                    )
                )
                break
            print(
                "当前历史最优解{}：{}".format(self.f_best, self.x_best)
            )  # 展示当前最优值
            print("当前接受的新解{}：{}".format(self.f, self.x))  # 展示当前接受的新解
            print("当前新解{}：{}".format(self.f_new, self.x_new))  # 展示当前新产生的解
            print("当前温度为{}".format(self.T))  # 展示当前温度

    # 当每个温度下的循环结束时，有一定概率将当前接受的新解替换为历史最优解
    def accept_best_xf(self):
        if np.random.random() > 0.75:
            self.x = self.x_best
            self.f = self.f_best

    def run(self):
        while self.T > self.T_min:
            self.run_T()  # 循环在该温度下的求解
            self.xf_best_T[self.T] = [
                self.get_history_best_xy()
            ]  # 记录在每一个温度下的最优解
            self.T_change()  # 温度继续下降
            self.accept_best_xf()  # 当每个温度下的循环结束时，有一定概率将当前接受的新解替换为历史最优解
            if self.condition_end() == True:  # 如果满足终止条件，终止该温度循环
                break



class NGHeston(NG):
    def __init__(self, func, x0):
        super().__init__(func, x0)
        self.T = 90#初始温度
        self.T_min = 1e-7  # 由于算法耗时太长，故小做一段模拟试试看
        self.times_max = 500#每个温度下循环次数

    # sv模型的各个参数由于存在取值范围，因此在获得新的参数估计值时需要对其取值范围加以限制
    def get_x_new(self):
        """
        [v0,kappa,theta,sigma,rho]
        其中：
        v0,kappa,theta,sigma>0
        -1<rho<1
        2kappa*theta>sigma**2
        """
        x = copy.deepcopy(self.x)  # 使用深copy，否则self.x会随着x一起变动
        x[0] = random_range(x[0], 0, 5)
        x[1] = random_range(x[1], 0, 1)
        x[2] = random_range(x[2], 0, 1)
        x[3] = random_range(x[3], 0, 3)
        x[4] = random_range(x[4], -1, 1)
        return x


class SV_SA:

    def __init__(
        self,
        data,
        v0: float = 0.01,
        kappa: float = 2,
        theta: float = 0.1,
        sigma: float = 0.1,
        rho: float = -0.5,
    ):
        """输入数据

        data    -pandas.core.frame.DataFrame格式数据，具体样式如下：

                           K         t        s0         r       c
                30     2.150  0.194444  2.111919  0.031060  0.0546
                31     2.150  0.198413  2.115158  0.031120  0.0666
                32     2.150  0.202381  2.107673  0.031210  0.0627
                33     2.150  0.214286  2.122269  0.031250  0.0531
                90     3.240  0.202381  3.181339  0.047446  0.0724

        """

        self.data = data
        self.init_params = [v0, kappa, theta, sigma, rho]  # 初始参数列表
        self.cycle = 0  # 计算模拟退火算法轮数
        self.error = 0.000000

    def error_mean_percent(self, init_params: list):
        """计算heston模型期权定价的百分比误差均值

        百分比误差均值=绝对值（（理论值-实际值）/实际值）/样本个数

        输入：
        init_params -初始参数,列表格式
                     [v0,kappa,theta,sigma,rho]

        返回： -误差百分点数   例如：返回5，表示5%
        """
        v0, kappa, theta, sigma, rho = init_params
        list_p_sv = []
        for i in self.data.index:
            K, t, s0, r, p_real = self.data.loc[i, :].tolist()
            sv = Heston_Model(
                K=K,
                t=t,
                S0=s0,
                r=r,
                v0=v0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                rho=rho,
            )
            p_sv = sv.Call_Value()  # sv模型期权价格
            list_p_sv.append(p_sv)

        self.error = np.average(
            np.abs((np.array(list_p_sv) - self.data["c"]) / self.data["c"])
        )  # sv模型的期权价格和实际价格的百分比误差均值
        print("\n")
        print("第{}轮,误差：{}".format(self.cycle, self.error))  # 展示本轮的误差
        self.cycle += 1

        return self.error

    def error_mean(self, init_params: list):
        """计算heston模型期权定价的均方误差
        init_params -初始参数,列表格式
                     [v0,kappa,theta,sigma,rho]
        """
        v0, kappa, theta, sigma, rho = init_params
        list_p_sv = []
        for i in self.data.index:
            K, t, s0, r, p_real = self.data.loc[i, :].tolist()
            sv = Heston_Model(
                K=K,
                t=t,
                S0=s0,
                r=r,
                v0=v0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                rho=rho,
            )
            p_sv = sv.Call_Value()  # sv模型期权价格
            list_p_sv.append(p_sv)

        self.error = np.sqrt(
            np.sum((np.array(list_p_sv) - self.data["c"]) ** 2) / len(self.data)
        )  # sv模型的期权价格和实际价格的均方误差
        print("\n")
        print("第{}轮,误差：{}".format(self.cycle, self.error))  # 展示本轮的误差
        self.cycle += 1

        return self.error

    def test_error_mean(self, multiple_parmas: dict):
        """将多组初始参数输入，计算各组参数的均方误差
        multiple_parmas -dict,多组初始参数
                        {
                        1:[0.01,2,0.1,0.1,-0.5],
                        2:[0.01,2,0.1,0.1,-0.5],
                        3:[0.01,2,0.1,0.1,-0.5]
                        }
        返回： -dict,记录各组初始参数的均方误差
        """
        dict_ = {}  # 用于记录各组初始参数的均方误差
        for i in multiple_parmas.keys():
            dict_[i] = self.error_mean(multiple_parmas[i])
        return dict_

    def test_option_price(self, multiple_parmas: dict):
        """将多组期权数据和初始参数输入，将期权价格合并在表格旁边
        multiple_parmas -dict,多组初始参数
                        multiple_parmas={
                        1:[1.5932492661058346, 3.3803420203705365, 0.3333248435472669, 5.622092726036617, 0.044881506437356666],
                        2:[1.1070063457234607, 3.501301312245266, 0.6276009140316863, 9.383112611111134, -0.6092511548040354],
                        3:[0.5675877305927083, 3.736229838972323, 0.21803303626214537, 8.74231319248172, 0.09393882921335006]
                        }
        返回： -dict,记录各组初始参数的均方误差
        """
        data_option_ = copy.deepcopy(self.data)
        for i in multiple_parmas.keys():
            # i=3
            data_option_["第{}组参数".format(i)] = 0.000000
            init_params = multiple_parmas[i]
            for j in data_option_.index:
                # j=8
                option_params = data_option_.loc[j, :4].tolist()
                sv = Heston_Model(
                    option_params[0],
                    option_params[1],
                    option_params[2],
                    option_params[3],
                    init_params[0],
                    init_params[1],
                    init_params[2],
                    init_params[3],
                    init_params[4],
                )
                c_sv = sv.Call_Value()
                data_option_.loc[j, "第{}组参数".format(i)] = c_sv
            print("已经完成第{}组".format(i))
        return data_option_

    def sa(self):
        """对均方误差函数用模拟退火算法计算最优值"""
        self.ng = NGHeston(func=self.error_mean_percent, x0=self.init_params)
        self.ng.run()
        self.x_star, self.y_star = self.ng.get_history_best_xy()

        print(self.x_star, self.y_star)  # 生成最优解x和最优值y


def getBestPara():
    data=pd.read_csv('上证50ETF期权数据.csv')#读取原始数据

    date_start_train=20200612#指定待训练数据的开始日期为2020年6月12日
    date_end_train=20200912#指定待训练数据的结束日期为2020年9月12日
    option = data[
        (data["交易日期"] >= date_start_train) & (data["交易日期"] <= date_end_train)
    ]  # 选择指定日期的数据

    option=option[option['看涨看跌类型']=='C']#训责看涨期权类型
    option=option[['执行价格','剩余到期时间（年）','上证50ETF价格','无风险收益率（shibor)','期权收盘价']]
    option.columns=['K','t','s0','r','c']#将数据列标题修改为指定列标题

    model=SV_SA(data=option)#建立类
    model.sa()#开始训练模型，不停地寻找最优解
    model.ng.get_history_best_xy()#查看最优解，即能使总误差最小的heston模型的五个参数

    return model.ng.get_history_best_xy()#返回最优解

if __name__ == "__main__":
    getBestPara()#调用函数