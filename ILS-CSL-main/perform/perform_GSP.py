import numpy as np
import pandas as pd

from hc.DAG import DAG
from hc.gsp import gsp_prior
from utils import parse_parents_score, soft_constraint, write_parents_score


def gsp_test(prefix="", prior_confidence=0.99999, **kwargs):
    '''
    使用GSP (Greedy Sparsest Permutation)算法学习贝叶斯网络结构
    :param prefix: 字符串，用于指定输出文件的前缀。
    :param prior_confidence: 浮点数，表示先验信息的置信度。
    :param **kwargs: 可变关键字参数，用于传递其他参数，如数据集名称（d）、样本集名称（s）、关系集名称（r）、存在的边（exist_edges）、禁止的边（forb_edges）等。
    :return: 学习到的贝叶斯网络的邻接矩阵。
    '''
    # 读取数据
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    # 根据数据类型决定dtype：'float64'为连续数据，'category'为离散数据
    data = pd.read_csv(filename, dtype='category')
    D = DAG(list(data.columns))
    
    # 加载数据和真实DAG结构
    D.load_data(data, test=None, score=None)
    D.load_truth_nptxt(f"BN_structure/{kwargs['d']}_graph.txt")

    exist_edges = kwargs['exist_edges']
    
    score_filepath = kwargs['score_filepath'] if 'score_filepath' in kwargs else None
    
    # 处理先验信息
    if kwargs["is_soft"]:
        # 软约束评分路径设置
        soft_score_path = f"data/score/tmp/{prefix}gsp_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.score"
        
        # 解析先前的分数
        score_dict = parse_parents_score(f"{score_filepath}")
        
        # 定义约束条件
        edge_cons, forb_cons = kwargs["exist_edges"], kwargs['forb_edges']
        soft_scorer = soft_constraint(obligatory=edge_cons, forbidden=forb_cons, 
                                     lamdba=[[prior_confidence for _ in range(len(edge_cons))],
                                             [prior_confidence for _ in range(len(forb_cons))]])
        
        # 更新分数字典
        for var in score_dict:
            for i, ls in enumerate(score_dict[var]):
                score, parent_set = ls
                prior_bonus = soft_scorer.calculate(var=int(var), parent=[int(p) for p in parent_set])
                new_score = score + prior_bonus
                score_dict[var][i] = (new_score, parent_set)
                
        # 写入更新后的分数
        write_parents_score(score_dict=score_dict, file_path=soft_score_path)
        
        # 更新分数文件路径
        score_filepath = soft_score_path
        
        # 初始化先验概率矩阵
        tmp = np.ones((D.data.shape[1], D.data.shape[1]))
        tmp[np.diag_indices_from(tmp)] = 0
        
        # 添加先验信息到DAG
        for prior in list(zip(*np.where(tmp))):
            D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])
    else:
        # 初始化先验概率矩阵
        tmp = np.ones((D.data.shape[1], D.data.shape[1]))
        
        # 处理禁止的边
        for prior in kwargs['forb_edges']:
            tmp[prior[0], prior[1]] = 0
            
        # 处理对角线元素
        tmp[np.diag_indices_from(tmp)] = 0
        
        # 处理必须存在的边
        for par, var in exist_edges:
            D.a_prior[D.varnames[var]]['par'].append(D.varnames[par])
            
        # 添加先验信息到DAG
        for prior in list(zip(*np.where(tmp))):
            D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])
            
    # 使用GSP算法学习BN结构
    gsp_prior(D, score_filepath=score_filepath)
    
    # 转换DAG为邻接矩阵
    D.dag2graph()
    
    # 返回邻接矩阵
    return D.graph 