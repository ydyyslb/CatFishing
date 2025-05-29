import numpy as np
import pandas as pd

from hc.DAG import DAG
from mmhc.mmhc import mmhc_prior

# 它用于运行基于 MMHC (Max-Min Hill-Climbing) 算法的因果发现测试
def mmhc_test(**kwargs):

    # 使用格式化字符串创建一个文件路径，该路径由数据集名称 d、样本大小 s 和随机种子 r 组成。
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"

    # 使用 pandas 库的 read_csv 函数读取 CSV 文件，并将其存储在变量 data 中。数据类型被设置为 category，这适用于分类数据；如果数据是连续的，可以将其更改为 dtype='float64'。
    data = pd.read_csv(filename, dtype='category') # change dtype = 'float64'/'category' if data is continuous/categorical
    # 创建一个 DAG 类的实例 D，并将数据列名作为参数传递给构造函数。
    D=DAG(list(data.columns))
    # 调用 DAG 类的 load_data 方法，将数据加载到 DAG 实例中。
    D.load_data(data,test=None,score=None)
    # 调用 DAG 类的 load_truth_nptxt 方法，加载真实的因果图结构。这里假设真实的因果图结构存储在一个文本文件中。
    D.load_truth_nptxt(f"BN_structure/{kwargs['d']}_graph.txt")
    #D.Add_prior(a_prior=edges)
    # 创建一个形状与数据列数相同的二维数组 tmp，并初始化所有元素为 1。
    tmp=np.ones((D.data.shape[1],D.data.shape[1]))
    # 禁止边的处理:
    # for prior in kwargs['forb_edges']:: 遍历 forb_edges 关键字参数提供的禁止边列表。
    # tmp[prior[0],prior[1]]=0: 在 tmp 数组中，将禁止边的对应位置设置为 0。
    # tmp[np.diag_indices_from(tmp)] = 0: 将对角线上的元素（表示变量与自身的边）设置为 0。
    for prior in kwargs['forb_edges']:
        tmp[prior[0],prior[1]]=0
    tmp[np.diag_indices_from(tmp)] = 0    
    # 添加先验知识:
    # for prior in list(zip(*np.where(tmp))):: 遍历 tmp 数组中为 1 的元素位置，即可能的边。
    # D.pc[D.varnames[prior[0]]].append(D.varnames[prior[1]]): 将这些可能的边作为先验知识添加到 DAG 实例的 pc 属性中。
    for prior in list(zip(*np.where(tmp))):
        D.pc[D.varnames[prior[0]]].append(D.varnames[prior[1]])

    # 调用 mmhc_prior 函数，使用 MMHC 算法基于先验知识和数据生成因果图。
    mmhc_prior(D)
    # 调用 DAG 类的 dag2graph 方法，将因果图转换为某种图形表示形式。
    D.dag2graph()
    # return D.graph: 返回生成的因果图。
    return D.graph
    
