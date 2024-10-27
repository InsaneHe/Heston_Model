在论文中，我参考了BSM模型的相关数学公式、以及Heston模型的模拟退火思路。

论文中使用了2019 年 12 月 23 日~2020 年 3 月 19 日 81 只沪深 300 股指期权数据  作为训练数据，2020 年 3 月 20 号至 2021 年 3 月 20 日的数据用于模型拟合优度检验。它得出的结论是：

+ BSM 模型中假设股价服从几何布朗运动过程  
+ Heston 模型不再假定波动率固定，而是认为其是随时间变动的一个变量，同时使用 5 个参数对波动率的变动情况进行模拟

论文得出的实验结论是：

+ 整体拟合优度：BSM模型 > Heston模型  
+ BSM模型对短到期期限、低价值的期权拟合效果相对较好， Heston模型对低执行价、短到期期限、低价值的期权拟合效果较好。  

但是在论文复现中，我们发现的结果是：

+ BSM模型的误差（0.476）>Heston模型的误差（0.339）
+ BSM模型误差率的变化率随时间变化曲线比Heston模型起伏变化更大

分析可能的原因在：

+ 训练数据与测试数据可能存在不同的市场环境、波动性或趋势，导致模型在新数据上的表现有所差异。如果测试数据中包含了极端价格或波动没有在训练集中得到充分体现，可能会导致BSM模型表现不佳。
+ BSM模型通常使用历史数据直接估计参数，比如波动率，而Heston模型可能采用更为复杂的参数估计方法，能够更好地拟合波动率的动态变化。如果BSM模型的参数估计不够精确，尤其是波动率的估计不稳定或滞后，可能导致其在新数据上的表现较差。
+ 市场中的非理性行为、流动性问题或其他结构性因素可能影响期权定价。



英语版：

In the paper, I referenced relevant mathematical formulas of the BSM model and the simulated annealing approach of the Heston model.

The paper used data from81 CSI300 index options from December23,2019, to March19,2020, as training data, while data from March20,2020, to March20,2021, was used for model goodness-of-fit testing. The conclusions drawn are:

- The BSM model assumes that stock prices follow a geometric Brownian motion process.
- The Heston model does not assume a fixed volatility; instead, it considers volatility as a variable that changes over time, simulating volatility changes with5 parameters.

The experimental conclusions from the paper are:

- Overall goodness-of-fit: BSM model > Heston model
- The BSM model fits better for short-term, low-value options, while the Heston model performs better for low strike price, short-term, low-value options.

However, in our replication of the paper, we found the following results:

- The error of the BSM model (0.476) > The error of the Heston model (0.339)
- The rate of change of the BSM model's error varies more significantly over time than that of the Heston model.

Possible reasons for these findings include:

- Differences in market conditions, volatility, or trends between the training and testing data could lead to differing model performance on new data. If the test data includes extreme prices or volatility not adequately represented in the training set, it may result in poor performance for the BSM model.
- The BSM model typically uses historical data to directly estimate parameters, such as volatility, while the Heston model may employ more complex parameter estimation methods that can better fit the dynamic changes in volatility. If the parameter estimation of the BSM model is not precise enough, especially if the volatility estimation is unstable or lagging, it may lead to poor performance on new data.
- Irrational behavior in the market, liquidity issues, or other structural factors may affect option pricing.
