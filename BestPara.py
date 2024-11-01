# 利用模拟退火算法寻找Heston模型最优参数（Finding Optimal Parameters of Heston Model Using Simulated Annealing Algorithm）
# 目标函数为：最小化市场价格和Heston模型价格的均方误差（The objective function is to minimize the mean square error between the market price and the Heston model price）

from Heston_Model import Heston_Model
import pandas as pd
import numpy as np
import copy



# 将x增加一个随机变动量，但会把x限制在a,b之内（Adds a random variation to x, but limits x to a,b）
def random_range(x, a, b):
    random = np.random.normal(0, 0.01 * (b - a), 1)[0]
    if x + random > b:
        random = -np.abs(random)  # 如果新值超越最大值，就将增量变为负数（If the new value exceeds the maximum value, make the increment negative）
    elif x + random < a:
        random = np.abs(random)  # 如果新值超越最小值，就将增量变为正数（If the new value exceeds the minimum value, make the increment positive）
    return x + random

# 模拟退火算法（Simulated Annealing Algorithms）
class NG:
    def __init__(self, func, x0):
        """
        x0  -列表：函数的初始值（List: Initial values of functions）
                [v0,kappa,theta,sigma,rho]
        func    -待求解函数（function to be solved）
        """
        self.x = x0
        self.dim_x = len(x0)  # 解的维度（Dimension of the solution）
        self.func = func
        self.f = func(self.x)  # 计算y值（Calculate the y-value）

        self.x_best = self.x  # 记录下来历史最优解，即所有循环中的最优解（Record the historical optimal solution, i.e. the optimal solution in all loops）
        self.f_best = self.f

        self.times_stay = 0  # 连续未接受新解的次数（Number of consecutive non-acceptance of new solutions）
        self.times_stay_max = 400  # 当连续未接受新解超过这个次数，终止循环（When the number of consecutive failures to accept new solutions exceeds this number, the loop is terminated）

        self.T = 100  # 初始温度：初始温度越高，前期接受新解的概率越大（Initial temperature: the higher the initial temperature, the higher the probability of accepting a new solution upfront）
        self.speed = 0.7  # 退火速度：温度下降速度越快，后期温度越低，接受新解的概率越小，从而最终稳定在某个解附件（Rate of annealing: the faster the rate of temperature decrease, the lower the temperature at a later stage and the lower the probability of accepting a new solution, thus eventually stabilizing at some solution attachment）
        self.T_min = 1e-6  # 最低温度：当低于该温度时，终止全部循环（Minimum temperature: terminates all cycles when below this temperature）
        self.xf_best_T = {}  # 记录下接受的所有新解（Record all new solutions accepted）

        # 最初若函数值变动为delta，则认为函数值变动很大，可以产生p_expec概率接受新解（Initially if the function value changes by delta, the function value is considered to have changed so much that a p_expec probability can be generated to accept the new solution）
        # 若在初期便产生巨大的概率接受新解，则前期寻找新解的过程将变成盲目的随机漫步毫无意义，因此利用alpha调节概率（If a huge probability of accepting a new solution is generated at an early stage, the process of finding a new solution in the early stages will become a blind random walk meaningless, so the probability is regulated using alpha）
        self.p_expec = 0.9
        self.delta_standard = 0.7
        self.alpha = self.find_alpha()  # 调节概率因子（moderating probability factor）

        self.times_delta_samller = 0  # 统计新旧最优值之差绝对值连续小于某值的次数（Counting the number of times the absolute value of the difference between the old and new optimal values is consecutively less than a certain value）
        self.delta_min = (
            0.001  # 当新旧最优值之差绝对值连续小于此值达到某一次数时，终止该温度循环（When the absolute value of the difference between the old and new optimal values is less than this value continuously for a certain number of times, the temperature cycle is terminated.）
        )
        self.times_delta_min = (
            100  # 当新旧最优值之差绝对值连续小于此值达到这个次数时，终止该温度循环（When the absolute value of the difference between the old and new optimal values is less than this value for this number of consecutive times, the temperature cycle is terminated.）
        )

        self.times_max = 500  # 当每个温度下循环超过这个次数，终止该温度循环（When this number of cycles per temperature is exceeded, the temperature cycle is terminated.）
        self.times_cycle = 0  # 记录算法循环总次数（Record the total number of algorithm loops）

        self.times_p = 0  # 统计因为p值较大而接受新解的次数（Count the number of times a new solution is accepted because of a large p-value）

        self.xf_all = {
            self.times_cycle: [self.x, self.f]
        }  # 记录下来每一次循环产生的新解和函数值（Record the new solution and function value generated by each loop）
        self.xf_best_all = {
            self.times_cycle: [self.x, self.f]
        }  # 记录下来每一次循环接受的新解和函数值（Record the new solution and function value accepted at each loop）

    # 温度下降，产生新温度（Temperature drops, generating new temperatures）
    def T_change(self):
        self.T = self.T * self.speed
        print(
            "当前温度为{},大于最小温度{}".format(self.T, self.T_min) # The current temperature is {},greater than the minimum temperature {}
        )  # 展示当前温度和最小温度（Display current and minimum temperatures）

    # 将所有的x和f、循环次数存储下来
    def save_xy(self):
        self.xf_all[self.times_cycle] = [self.x, self.f]

    # 将所有的最优x,y、循环次数存储下来（Store all the x's and f's, the number of loops）
    def save_best_xy(self):
        self.xf_best_all[self.times_cycle] = [self.x, self.f]

    # 当调节因子为alpha时，函数值变动值为delta产生的接受新解概率（When the moderator is alpha, the function value change value is the probability of accepting a new solution produced by delta）
    def __p_delta(self, alpha):
        return np.exp(-self.delta_standard / (self.T * alpha))

    # 用二分法寻找方程解（Finding solutions to equations by dichotomization）
    def __find_solver(self, func, f0):
        """
        输入：（Input:）
        func    -待求解方程的函数（Functions of the equation to be solved）
        f0  -float,预期函数值（Expected function value）
        输出：（Output:）
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

    # 最初若函数值变动为delta，则认为函数值变动很大，可以产生p_expec概率接受新解（Initially if the function value changes by delta, the function value is considered to have changed so much that a p_expec probability can be generated to accept the new solution）
    def find_alpha(self):
        return self.__find_solver(self.__p_delta, self.p_expec)

    # 获得新的x（Get a new x）
    def get_x_new(self):
        random = np.random.normal(0, 1, self.dim_x)  # 新的随机增量（a new randomized increment）
        return self.x + random

    # 判断是否可以接受新的解（Determine if a new solution is acceptable）
    def judge(self):
        if self.delta < 0:  # 如果函数值变动幅度小于0,则接受新解（If the function value changes by less than 0, the new solution is accepted.）
            self.x = self.x_new
            self.f_last = self.f  # 在最优解函数值更新之前将其记录下来（Record the value of the optimal solution function before it is updated）
            self.f = self.f_new
            self.save_best_xy()  # 记录每次循环接受的新解（Record the new solutions accepted in each loop）
            self.get_history_best_xy()  # 更新历史最优解（Update the historical optimal solution）
            self.times_stay = 0  # 由于未接受新解，将连续未接受新解的次数归零（Zero out the number of consecutive non-acceptance of new solutions due to non-acceptance of new solutions）
            print(
                "由于函数值变小新接受解{}:{}".format(self.f, self.x) # As the function value becomes smaller， the new accepted solution {}:{}
            )  # 展示当前接受的新解（Demonstration of currently accepted new solutions）
        else:
            p = np.exp(-self.delta / (self.T * self.alpha))  # 接受新解的概率（Probability of accepting a new solution）
            p_ = np.random.random()  # 判断标准概率
            if p > p_:  # 如果概率足够大，接受新解（If the probability is large enough to accept the new solution）
                self.x = self.x_new
                self.f_last = self.f  # 在接受的新解更新之前将其记录下来（Documentation of accepted new solutions before they are updated）
                self.f = self.f_new
                self.save_best_xy()  # 记录每次循环接受的新解（Record the new solutions accepted in each loop）
                self.get_history_best_xy()  # 更新历史最优解（Update the historical optimal solution）
                print(
                    "由于概率{}大于{}，新接受解{}:{}".format(p, p_, self.f, self.x) # Since the probability {} is greater than {}, the new accepted solution {}:{}
                )  # 展示当前接受的新解（Demonstration of currently accepted new solutions）
                self.times_p += 1  # 统计因为概率而接受新解的次数（Count the number of times a new solution is accepted because of the probability）
                self.times_stay = 0  # 由于未接受新解，将连续未接受新解的次数归零（Zero out the number of consecutive non-acceptance of new solutions due to non-acceptance of new solutions）
            else:
                if self.time_ == 0:
                    self.f_last = self.f  # 在接受的新解更新之前将其记录下来（Documentation of accepted new solutions before they are updated）
                self.times_stay += 1  # 连续接受新解次数加1（Add 1 to the number of consecutive new interpretations）
                print("连续未接受新解{}次".format(self.times_stay)) # No new solutions accepted {} times in a row

    # 获得历史最优解（Obtaining historically optimal solutions）
    def get_history_best_xy(self):
        x_array = list(
            np.array(list(self.xf_best_all.values()),dtype=object)[:, 0]
        )  # 从历史所有的最优x和f中获得所有的x（Obtain all x from the history of all optimal x and f）
        f_array = list(
            np.array(list(self.xf_best_all.values()),dtype=object)[:, 1]
        )  # 从历史所有的最优x和f中获得所有的f（Obtain all f from all optimal x and f in history）
        self.f_best = min(f_array)  # 从每阶段最优的f中获得最优的f（Obtain the optimal f from the optimal f at each stage）
        self.x_best = x_array[f_array.index(self.f_best)]  # 利用最优f反推最优x（Use the optimal f to invert the optimal x）
        return self.x_best, self.f_best


    # 统计新旧函数值之差的绝对值连续小于此值的次数（Count the number of times the absolute value of the difference between the old and new function values is consecutively less than this value）
    def count_times_delta_smaller(self):
        if self.delta_best < self.delta_min:
            self.times_delta_samller += (
                1  # 如果新旧函数值之差绝对值小于某值，则次数加1，否则归零（If the absolute value of the difference between the old and new function values is less than a certain value, the number of times is increased by 1, otherwise it goes to zero）
            )
        else:
            self.times_delta_samller = 0
        print(
            "差值{}连续小于{}达到{}次".format( # The difference {} is less than {} up to {} times in a row.
                self.delta_best, self.delta_min, self.times_delta_samller
            )
        )

    # 终止循环条件（Termination conditions）
    def condition_end(self):
        if (
            self.times_delta_samller > self.times_delta_min
        ):  # 如果新旧函数值之差绝对值连续小于某值次数超过某值，终止该温度循环（If the absolute value of the difference between the old and new function values is less than a certain value for more than a certain number of times in a row, the temperature cycle is terminated.）
            return True
        elif (
            self.times_stay > self.times_stay_max
        ):  # 当连续未接受新解超过这个次数，终止循环（When the number of consecutive failures to accept new solutions exceeds this number, the loop is terminated）
            return True

    # 在某一特定温度下进行循环（Cycling at a specific temperature）
    def run_T(self):

        for time_ in range(self.times_max):
            self.time_ = time_
            self.x_new = self.get_x_new()  # 获得新解（Acquisition of new solutions）
            self.f_new = self.func(self.x_new)  # 获得新的函数值（Getting a new function value）
            self.save_xy()  # 将新解和函数值记录下来（Record the new solution and function values）
            self.delta = self.f_new - self.f  # 计算函数值的变化值（Calculate the change in value of the function）
            self.judge()  # 判断是否接受新解（Determine whether to accept the new solution）
            self.times_cycle += 1  # 统计循环次数（Counting the number of loops）
            self.delta_best = np.abs(
                self.f - self.f_last
            )  # 上次函数值与这次函数值的差值绝对值（Absolute value of the difference between the last function value and this function value）
            self.count_times_delta_smaller()  # 统计新旧函数值之差的绝对值连续小于此值的次数（Count the number of times the absolute value of the difference between the old and new function values is consecutively less than this value）
            if self.condition_end() == True:  # 如果满足终止条件，终止该温度循环（If the termination condition is met, terminate this temperature cycle）
                print(
                    "满足终止条件：接受新解后的函数值变化连续小于{}达到次数".format( # Satisfy the termination condition: the change in the value of the function after accepting the new solution is consecutively less than {} up to a number of times
                        self.delta_min
                    )
                )
                break
            print(
                "当前历史最优解{}：{}".format(self.f_best, self.x_best) # Current historical optimal solution {}:{}
            )  # 展示当前最优值（Display the current optimal value）
            print("当前接受的新解{}：{}".format(self.f, self.x))  # 展示当前接受的新解（Demonstrate the currently accepted new solutions）
            print("当前新解{}：{}".format(self.f_new, self.x_new))  # 展示当前新产生的解（Demonstrate the current generation of new solutions）
            print("当前温度为{}".format(self.T))  # 展示当前温度（Demonstrate current temperature）

    # 当每个温度下的循环结束时，有一定概率将当前接受的新解替换为历史最优解（When the cycle ends at each temperature, there is a certain probability that the currently accepted new solution will be replaced by the historically optimal solution）
    def accept_best_xf(self):
        if np.random.random() > 0.75:
            self.x = self.x_best
            self.f = self.f_best

    def run(self):
        while self.T > self.T_min:
            self.run_T()  # 循环在该温度下的求解
            self.xf_best_T[self.T] = [
                self.get_history_best_xy()
            ]  # 记录在每一个温度下的最优解（Record the optimal solution at each temperature）
            self.T_change()  # 温度继续下降（The temperature continues to drop）
            self.accept_best_xf()  # 当每个温度下的循环结束时，有一定概率将当前接受的新解替换为历史最优解（When the cycle ends at each temperature, there is a certain probability that the currently accepted new solution will be replaced by the historically optimal solution）
            if self.condition_end() == True:  # 如果满足终止条件，终止该温度循环（If the termination condition is met, terminate this temperature cycle）
                break



class NGHeston(NG):
    def __init__(self, func, x0):
        super().__init__(func, x0)
        self.T = 90#初始温度（initial temperature）
        self.T_min = 1e-7  # 最低温度：由于算法耗时太长，故小做一段模拟试试看（Minimum temperature: Because the algorithm is too time-consuming, do a small simulation to try to see）
        self.times_max = 500#每个温度下循环次数（Number of cycles per temperature）

    # sv模型的各个参数由于存在取值范围，因此在获得新的参数估计值时需要对其取值范围加以限制（The individual parameters of the sv model need to be constrained in obtaining new parameter estimates due to the existence of their range of values）
    def get_x_new(self):
        """
        [v0,kappa,theta,sigma,rho]
        其中：
        v0,kappa,theta,sigma>0
        -1<rho<1
        2kappa*theta>sigma**2
        """
        x = copy.deepcopy(self.x)  # 使用深copy，否则self.x会随着x一起变动（Use deepcopy, otherwise self.x will change with x）
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
        """输入数据（input data）

        data    -pandas.core.frame.DataFrame格式数据，具体样式如下：（pandas.core.frame.DataFrame format data in the following style:）

                           K         t        s0         r       c
                30     2.150  0.194444  2.111919  0.031060  0.0546
                31     2.150  0.198413  2.115158  0.031120  0.0666
                32     2.150  0.202381  2.107673  0.031210  0.0627
                33     2.150  0.214286  2.122269  0.031250  0.0531
                90     3.240  0.202381  3.181339  0.047446  0.0724

        """

        self.data = data
        self.init_params = [v0, kappa, theta, sigma, rho]  # 初始参数列表（Initial parameter list）
        self.cycle = 0  # 计算模拟退火算法轮数（Calculate the number of rounds of the simulated annealing algorithm）
        self.error = 0.000000

    def error_mean_percent(self, init_params: list):
        """计算heston模型期权定价的百分比误差均值（Calculating the Mean of Percentage Errors in Option Pricing for the heston Model）

        百分比误差均值=绝对值（（理论值-实际值）/实际值）/样本个数（Percentage error mean = absolute value ((theoretical value - actual value)/actual value)/number of samples）

        输入：（Input:）
        init_params -初始参数,列表格式（Initial parameters, list format）
                     [v0,kappa,theta,sigma,rho]

        返回： -误差百分点数   例如：返回5，表示5%（ Return: - Percentage points of error e.g. Returns 5, which means 5%.）
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
            p_sv = sv.Call_Value()  # sv模型期权价格（sv model option prices）
            list_p_sv.append(p_sv)

        self.error = np.average(
            np.abs((np.array(list_p_sv) - self.data["c"]) / self.data["c"])
        )  # sv模型的期权价格和实际价格的百分比误差均值（Mean of percentage error between option price and actual price for the sv model）
        print("\n")
        print("第{}轮,误差：{}".format(self.cycle, self.error))  # 展示本轮的误差（Demonstrate the error in this round）
        self.cycle += 1

        return self.error



    def sa(self):
        """对均方误差函数用模拟退火算法计算最优值（Calculation of the optimal value using simulated annealing algorithm for the mean square error function）"""
        self.ng = NGHeston(func=self.error_mean_percent, x0=self.init_params)
        self.ng.run()
        self.x_star, self.y_star = self.ng.get_history_best_xy()

        print(self.x_star, self.y_star)  # 生成最优解x和最优值y（Generate optimal solution x and optimal value y）


def getBestPara():
    data=pd.read_csv('上证50ETF期权数据.csv')#读取原始数据：上证50ETF期权数据.csv（Read raw data：SSE50ETF Options Data.csv）

    date_start_train=20240919#选择训练数据的开始日期（Select the start date of the training data）
    date_end_train=20241019#选择训练数据的结束日期（Select the end date of the training data）
    option = data[
        (data["交易日期"] >= date_start_train) & (data["交易日期"] <= date_end_train)# Date of transaction
    ]  # 选择指定日期的数据（Selecting data for a specified date）

    option=option[option['看涨看跌类型']=='C']#训责看涨期权类型（Types of Call Options）
    option=option[['执行价格','剩余到期时间（年）','上证50ETF价格','无风险收益率（shibor)','期权收盘价']]#（'Exercise price','Remaining maturity (years)','SSE 50 ETF price','Risk-free yield (shibor)','Option closing price'）
    option.columns=['K','t','s0','r','c']#将数据列标题修改为指定列标题（Change data column headers to specified column headers）

    model=SV_SA(data=option)#建立类（creating a class）
    model.sa()#开始训练模型，不停地寻找最优解（Start training the model and keep looking for the optimal solution）
    model.ng.get_history_best_xy()#查看最优解，即能使总误差最小的heston模型的五个参数（View the five parameters of the optimal solution, i.e., the heston model that minimizes the total error）

    return model.ng.get_history_best_xy()#返回最优解（Return the optimal solution）

if __name__ == "__main__":
    getBestPara()#调用函数（call the function）
