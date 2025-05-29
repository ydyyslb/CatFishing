# 因果发现算法

本模块包含多种因果发现算法的实现，包括HC (Hill Climbing)和GSP (Greedy Sparsest Permutation)。

## HC算法

HC算法是一种基于分数的贪婪搜索算法，通过不断尝试添加、删除或反转边来优化DAG结构，直到无法进一步提高分数。

## GSP算法 

GSP算法是一种基于排序的因果发现方法，它首先通过贪婪方式找到变量的最优排序，然后为每个变量选择最优的父节点集。

### 特点

- 不容易陷入局部最优解
- 在处理大规模网络时效率更高
- 在存在隐变量的情况下通常表现更好
- 支持先验知识的整合

### 使用方法

#### 基本使用

```python
from hc.gsp import gsp
import pandas as pd

# 准备数据
data = pd.read_csv('your_data.csv')
varnames = list(data.columns)

# 离散数据
arities = data.nunique().values  # 每个变量的不同取值数量
result = gsp(data.values, arities, varnames, score='bic')

# 连续数据
result = gsp(data.values, None, varnames, score='bic_g')

# 输出结果
for var in result:
    print(f"{var}: {result[var]['par']}")
```

#### 使用先验知识

```python
from hc.gsp import gsp_prior
from hc.DAG import DAG
import pandas as pd

# 准备数据
data = pd.read_csv('your_data.csv')
varnames = list(data.columns)

# 创建DAG对象
D = DAG(varnames)
D.load_data(data)

# 添加先验知识
# 已知的因果关系
D.a_prior['Y']['par'] = ['X1']  # X1 -> Y

# 禁止的因果关系
D.d_prior['X1']['par'] = ['Y']  # 禁止 Y -> X1

# 运行带先验知识的GSP
result = gsp_prior(D)

# 输出结果
for var in result:
    print(f"{var}: {result[var]['par']}")
```

## 测试

测试文件 `test_gsp.py` 包含示例代码，展示如何使用GSP算法并与真实DAG进行比较：

```
python test_gsp.py
```

## 参考文献

- Solus, L., Wang, Y., Matejovicova, L., & Uhler, C. (2017). Consistency guarantees for permutation-based causal inference algorithms.
- Wang, Y., & Drton, M. (2020). High-dimensional causal discovery under non-Gaussianity. 