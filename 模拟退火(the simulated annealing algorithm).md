函数关系图

```
getBestPara
    └── SV_SA
        ├── error_mean_percent
        ├── error_mean
        ├── test_error_mean
        ├── test_option_price
        └── sa
            └── NGHeston (继承自 NG)
                ├── get_x_new (重写)
                └── run (继承自 NG)
                    ├── run_T
                    ├── T_change
                    ├── save_xy
                    ├── save_best_xy
                    ├── __p_delta
                    ├── __find_solver
                    ├── find_alpha
                    ├── judge
                    ├── get_history_best_xy
                    ├── count_times_delta_smaller
                    ├── condition_end
                    └── accept_best_xf
```

对于NG类：该类是基础的模拟退火算法，用于优化目标函数

该类的执行步骤如下：

+ 首先执行初始化函数`def __init__(self, func, x0)`。在进行模拟退火前，首先要提供优化参数的初始值，模拟退火会在这个初始值的基础上进行微调，在这里初始值即`x0`。同时也要传递待优化的函数`func`。此外，该初始化函数中还需要设置其他参数，在代码中有注释。

+ 正式进入模拟退火后，会调用`def run(self)`方法，开始模拟退火算法的主循环。

  + 循环条件为`while self.T > self.T_min`，即当前温度高于设定的最低温度。
  + 在`self.T`下，执行`run_T(self)`方法，该方法通过多次迭代来寻找函数的最优解。每次迭代开始时，生成一个新的解 `x_new`，并计算其对应的函数值 `f_new`。然后，将新解和函数值记录下来，并计算当前函数值与上一次函数值的差值 `delta`。
  + 接下来，通过调用 `judge` 方法来决定是否接受新解。如果接受新解，则更新当前解和函数值，并记录最优解。每次循环结束时，更新循环次数 `times_cycle`，并计算当前函数值与上一次函数值的绝对差值 `delta_best`，然后统计新旧函数值之差的绝对值连续小于某个阈值的次数。
  + 如果满足终止条件 `condition_end`，则打印终止信息并退出循环。否则，打印当前历史最优解、当前接受的新解、当前新产生的解以及当前温度的信息

+ 完成一次在该温度下的循环后，`      self.xf_best_T[self.T] = [self.get_history_best_xy() ]`将当前温度 `self.T` 下的最优解记录在 `self.xf_best_T`字典中。

  关于函数`get_history_best_xy`：`self.xf_best_all`是一个字典，存储了所有历史最优解的 x 和 f 值。`x_array`是一个列表，包含了从 `self.xf_best_all` 字典中提取的所有历史最优解的 x 值；`f_array`同理，包含所有历史最优解的 f 值。最后返回的`self.f_best`来自`f_array`中的最小值，`self.x_best`来自`min(f_array)`在`x_array`中的反推结果。

+ 现在，在`self.T`温度下的循环完成了，此时应该要降低温度，调用`self.T_change()`方法后，在原温度上乘上一个常数`self.speed`，这个常数在`def __init__(self, func, x0)`中设定，表征降温速度。

+ 当然，在每个温度循环结束后，我们也不一定把该温度下得到的`self.f_best`和`self.x_best`作为真正的最优解。我们会调用`self.accept_best_xf() `。通过生成一个随机数，并与0.75进行比较。如果随机数大于0.75，则将当前解 `x` 和函数值 `f` 替换为历史最优解 `x_best` 和 `f_best`。这种机制引入了一定的随机性，确保算法在局部最优解附近进行充分探索，同时也有助于跳出局部最优解，向全局最优解靠近。

+ 最后，为了提升算法效率，加入了提前循环停止的条件`self.condition_end()`。

  + **新旧函数值之差绝对值连续小于某值且次数超过某值**：如果新旧函数值之差的绝对值连续小于某个阈值的次数超过了预定义的最小次数，则返回 `True`，表示满足终止条件。
  + **连续未接受新解超过某个次数**：如果连续未接受新解的次数超过了预定义的最大次数，则返回 `True`，表示满足终止条件。

对于类NGHeston：重用 `NG`类中已有的模拟退火算法的实现，同时根据 Heston 模型的特定需求进行定制。

+ 函数`def __init__(self, func, x0)`：为Heston 模型的模拟退火设定特殊的初始温度、最低温度、循环次数等。
+ 函数`def get_x_new(self)`：Heston 模型中各参数有取值范围，用于替换NG类的`def get_x_new(self)`，在`run_T`取新的`self.x_new`时能加以限制。

对于类SV_SA：利用类NGHeston来执行模拟退火算法。

+ 在函数`def sa(self)`中，首先定义待优化函数`def error_mean_percent(self, init_params: list)`，以及参数的初始值`self.init_params`，建立`NGHeston`对象。
+ 随后调用`run()`和`get_history_best_xy()`来获取最优解。







英语版

For the NG class: This class is the fundamental simulated annealing algorithm used to optimize the objective function.

The execution steps of this class are as follows:

- First, the initialization function `def __init__(self, func, x0)` is executed. Before performing simulated annealing, it is necessary to provide the initial values of the parameters to be optimized; simulated annealing will fine-tune based on this initial value, which here is `x0`. The function to be optimized, `func`, should also be passed. Additionally, other parameters need to be set within this initialization function, which are commented in the code.

- Once in the simulated annealing process, the `def run(self)` method is called, starting the main loop of the simulated annealing algorithm.

  + The loop condition is `while self.T > self.T_min`, meaning that the current temperature must be higher than the set minimum temperature.
  + Under the temperature `self.T`, the method `run_T(self)` is executed, which seeks the optimal solution of the function through multiple iterations. At the start of each iteration, a new solution `x_new` is generated, and its corresponding function value `f_new` is calculated. The new solution and function value are then recorded, and the difference `delta` between the current function value and the last function value is calculated.
  + Next, the `judge` method is called to determine whether to accept the new solution. If the new solution is accepted, the current solution and function value are updated, and the optimal solution is recorded. At the end of each loop iteration, the count of iterations `times_cycle` is updated, and the absolute difference `delta_best` between the current function value and the last function value is calculated. Additionally, the number of times the absolute difference between the new and old function values is continuously smaller than a certain threshold is counted.
  + If the termination condition `condition_end` is satisfied, a termination message is printed, and the loop exits. Otherwise, information about the current historical optimal solution, the newly accepted solution, the newly generated solution, and the current temperature is printed.

- After completing one loop at the current temperature, `self.xf_best_T[self.T] = [self.get_history_best_xy()]` records the optimal solution at the current temperature `self.T` in the `self.xf_best_T` dictionary.

  Regarding the function `get_history_best_xy`: `self.xf_best_all` is a dictionary that stores all historical optimal solutions' x and f values. `x_array` is a list containing all x values extracted from the `self.xf_best_all` dictionary for historical optimal solutions; `f_array` similarly contains all f values of historical optimal solutions. The returned `self.f_best` comes from the minimum value in `f_array`, while `self.x_best` comes from back-calculating the result of `min(f_array)` within `x_array`.

- Now that the loop at temperature `self.T` is complete, it is time to lower the temperature. After calling `self.T_change()`, the original temperature is multiplied by a constant `self.speed`, which is set in `def __init__(self, func, x0)` to represent the cooling speed.

- Of course, after each temperature cycle, we do not necessarily take the `self.f_best` and `self.x_best` obtained at that temperature as the true optimal solution. We will call `self.accept_best_xf()`. By generating a random number and comparing it with 0.75, if the random number is greater than0.75, the current solution `x` and function value `f` will be replaced with the historical optimal solution `x_best` and `f_best`. This mechanism introduces a certain randomness, ensuring that the algorithm sufficiently explores around local optimal solutions while also helping to escape local optima towards global optima.

- Finally, to enhance the efficiency of the algorithm, a condition for early loop termination has been added, `self.condition_end()`.

  + **The absolute difference between new and old function values is continuously smaller than a certain value, and the number of occurrences exceeds a certain threshold**: If the number of times the absolute difference between new and old function values is continuously less than a certain threshold exceeds a predefined minimum number, it returns `True`, indicating that the termination condition is met.

  + **Consecutive times without accepting a new solution exceed a certain number**: If the number of consecutive times without accepting a new solution exceeds a predefined maximum number, it returns `True`, indicating that the termination condition is met.

For the NGHeston class: It reuses the implementation of the simulated annealing algorithm in the NG class while customizing it according to the specific requirements of the Heston model.

- Function `def __init__(self, func, x0)`: Sets special initial temperature, minimum temperature, number of cycles, etc., for the simulated annealing of the Heston model.
- Function `def get_x_new(self)`: The parameters of the Heston model have value ranges to replace `def get_x_new(self)` in the NG class, imposing restrictions when obtaining the new `self.x_new` in `run_T`.

For the SV_SA class: It utilizes the NGHeston class to execute the simulated annealing algorithm.

- In the function `def sa(self)`, the function to be optimized `def error_mean_percent(self, init_params: list)` is first defined, along with the initial values of the parameters `self.init_params`, and an `NGHeston` object is established.
- Then, it calls `run()` and `get_history_best_xy()` to obtain the optimal solution.
