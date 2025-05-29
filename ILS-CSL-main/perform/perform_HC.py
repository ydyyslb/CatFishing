import numpy as np
import pandas as pd

from hc.DAG import DAG
from hc.hc import hc_prior
from utils import parse_parents_score, soft_constraint, write_parents_score


# 它用于测试使用启发式合并（Hill Climbing）算法学习贝叶斯网络（Bayesian Network, BN）结构的过程。
# prefix: 字符串，用于指定输出文件的前缀。
# prior_confidence: 浮点数，表示先验信息的置信度。
# **kwargs: 可变关键字参数，用于传递其他参数，如数据集名称（d）、样本集名称（s）、关系集名称（r）、存在的边（exist_edges）、禁止的边（forb_edges）等。
def hc_test(prefix="",prior_confidence=0.99999, **kwargs):
    # 读取数据：
    # 从指定路径读取CSV文件，并将其数据类型设置为category。
    # 创建一个DAG对象，其列对应于数据集的列。
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    # change dtype = 'float64'/'category' if data is continuous/categorical
    data = pd.read_csv(filename, dtype='category')
    D = DAG(list(data.columns))
    # 加载数据和真实DAG结构：
    # 使用load_data方法加载数据。确定test和score的方法
    # 使用load_truth_nptxt方法加载真实DAG结构。
    D.load_data(data, test=None, score=None)
    D.load_truth_nptxt(f"BN_structure/{kwargs['d']}_graph.txt")


    exist_edges = kwargs['exist_edges']

    score_filepath = kwargs['score_filepath'] if 'score_filepath' in kwargs else None
    # 处理先验信息：
    # 如果is_soft参数为True，则创建一个软约束评分函数，并更新分数字典。
    # 否则，创建一个硬约束评分函数，并添加存在的边和禁止的边到DAG对象中。
    if kwargs["is_soft"]:
        # 软约束评分路径设置
        soft_score_path = f"data/score/tmp/{prefix}hc_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.score"
        # 解析先前的分数
        score_dict = parse_parents_score(f"{score_filepath}")
        # 定义约束条件
        # 这里定义了必须存在的边（obligatory）和禁止存在的边（forbidden）。
        # soft_constraint对象用于计算软约束下的先验信息分数。lamdba参数是一个列表，它为每个约束条件指定了置信度。
        edge_cons, forb_cons = kwargs["exist_edges"], kwargs['forb_edges']
        soft_scorer = soft_constraint(obligatory=edge_cons, forbidden=forb_cons, lamdba=[[prior_confidence for _ in range(len(edge_cons))],[prior_confidence for _ in range(len(forb_cons))]])
        # 更新分数字典
        # 对于分数字典中的每个变量，该循环遍历其所有可能的父集组合
        for var in score_dict:
            for i, ls in enumerate(score_dict[var]):
                score, parent_set = ls
                # soft_scorer.calculate用于计算给定变量和其父集组合的软约束先验分数。然后，将这个先验分数加到原始分数上，并更新分数字典。
                prior_bonus = soft_scorer.calculate(var=int(var),parent=[int(p) for p in parent_set])
                new_score = score + prior_bonus
                score_dict[var][i] = (new_score, parent_set)
        # 写入更新后的分数
        write_parents_score(score_dict=score_dict, file_path=soft_score_path)
        # 更新分数文件路径
        score_filepath = soft_score_path
        # 初始化先验概率矩阵
        # 创建一个和变量数量一样大小的矩阵，并将对角线元素设置为0
        tmp = np.ones((D.data.shape[1], D.data.shape[1]))
        tmp[np.diag_indices_from(tmp)] = 0
        # 添加先验信息到DAG
        # 对于矩阵tmp中的每个非零元素，这行代码将对应的变量关系添加到DAG对象的先验条件概率（pc）列表中。这意味着任何非零元素tmp[i, j] = 1表示变量j可能是变量i的父变量。
        for prior in list(zip(*np.where(tmp))):
            D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])
    else:
        # 初始化先验概率矩阵
        # 创建一个与数据集中变量数量相同的矩阵，初始时矩阵中的所有元素都设置为1，表示所有变量之间都可能存在关系。
        tmp = np.ones((D.data.shape[1], D.data.shape[1]))
        # 处理禁止的边:
        # 对于每一对禁止的边（forb_edges），将它们在矩阵tmp中的对应位置设置为0，表示这些边在最终的贝叶斯网络结构中不应该存在。
        for prior in kwargs['forb_edges']:
            tmp[prior[0], prior[1]] = 0
        # 处理对角线元素
        # 将矩阵tmp的对角线元素设置为0，因为一个变量不能是自己的父变量。
        tmp[np.diag_indices_from(tmp)] = 0
        # 处理必须存在的边:
        # 对于每一对必须存在的边（exist_edges），将它们添加到DAG对象的a_prior属性中。
        # a_prior是一个字典，其中包含了每个变量的先验父变量信息。这里，par是父变量的索引，var是子变量的索引，而D.varnames是一个将索引映射到变量名的列表。
        for par, var in exist_edges:
            D.a_prior[D.varnames[var]]['par'].append(D.varnames[par])
        # 添加先验信息到DAG
        # 对于矩阵tmp中的每个非零元素，这行代码将对应的变量关系添加到DAG对象的先验条件概率（pc）列表中。这意味着任何非零元素tmp[i, j] = 1表示变量j可能是变量i的父变量
        for prior in list(zip(*np.where(tmp))):
            D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])

    # 使用启发式合并算法学习BN结构：
    # 调用hc_prior函数，使用更新后的评分函数和先验信息学习BN结构。
    hc_prior(D, score_filepath=score_filepath)
    # 转换DAG为邻接矩阵：
    # 使用dag2graph方法将DAG转换为邻接矩阵。
    D.dag2graph()
    # 返回邻接矩阵：
    # 返回学习到的BN的邻接矩阵。
    return D.graph