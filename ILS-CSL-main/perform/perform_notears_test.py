import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from notears.notears.nonlinear import NotearsMLP, notears_nonlinear_dict
from hc.DAG import DAG
from utils import parse_parents_score, soft_constraint, write_parents_score
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
def calculate_shd(W_binary, truth_graph):
    """计算结构汉明距离（SHD）"""
    shd = np.sum(np.abs(W_binary - truth_graph))
    return shd
def find_best_threshold(W_est, truth_graph):
    thresholds = np.linspace(W_est.min(), W_est.max(), num=100)
    best_threshold = thresholds[0]
    best_score = float('inf')  # 对于SHD，分数越低越好

    for threshold in thresholds:
        W_binary = (W_est > threshold).astype(int)
        score = calculate_shd(W_binary, truth_graph)  # 计算SHD或其他评估指标
        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def no_test(i2p, prior_confidence=0.99999, prefix="", **kwargs):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # 读取数据
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    print(f"Loading data from {filename}")
    data = pd.read_csv(filename, dtype='category')
    D = DAG(list(data.columns))  # 使用hc的DAG类
    D.load_data(data, test=None, score=None)
    D.load_truth_nptxt(f"BN_structure/{kwargs['d']}_graph.txt")  # 加载真值图

    # 处理先验信息
    edge_cons, forb_cons = kwargs.get("exist_edges", []), kwargs.get("forb_edges", [])
    print(f"Existing edges: {edge_cons}, Forbidden edges: {forb_cons}")

    # 标签编码非数值列
    struct_data = data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
    for col in non_numeric_columns:
        le = LabelEncoder()
        struct_data[col] = le.fit_transform(struct_data[col])

    # 转换为NumPy数组并标准化
    X_np = struct_data.to_numpy()
    X_np = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)
    print(X_np)
    # 解析评分字典
    score_filepath = kwargs.get('score_filepath', None)
    score_dict = parse_parents_score(f"{score_filepath}") if score_filepath else {}
    print(f"Loaded score dictionary from {score_filepath}")

    soft_scorer = soft_constraint(
        obligatory=edge_cons,
        forbidden=forb_cons,
        lamdba=[[prior_confidence for _ in range(len(edge_cons))],
                [prior_confidence for _ in range(len(forb_cons))]]
    )

    # 更新分数字典
    for var in score_dict:
        for i, ls in enumerate(score_dict[var]):
            score, parent_set = ls
            prior_bonus = soft_scorer.calculate(var=int(var), parent=[int(p) for p in parent_set])
            score_dict[var][i] = (prior_bonus, parent_set)

    # 初始化先验概率矩阵并更新 DAG
    tmp = np.ones((D.data.shape[1], D.data.shape[1]))
    tmp[np.diag_indices_from(tmp)] = 0
    for prior in list(zip(*np.where(tmp))):
        D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])

    # 写入更新后的分数
    soft_score_path = f"data/score/tmp/{prefix}no_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.score"
    os.makedirs(os.path.dirname(soft_score_path), exist_ok=True)
    write_parents_score(score_dict=score_dict, file_path=soft_score_path)
    score_filepath = soft_score_path  # 更新分数文件路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义模型
    model = NotearsMLP(dims=[X_np.shape[1], 10, 1], bias=True).to(device)
    print("Model initialized.")

    # 运行 NOTears 算法
    W_est = notears_nonlinear_dict(model, X_np, D, score_filepath=score_filepath, lambda1=0.001, lambda2=0.001)

    # 寻找最佳阈值
    best_threshold = find_best_threshold(W_est, D.truth_graph)
    W_est_thresholded = (W_est > best_threshold).astype(int)
    return W_est_thresholded