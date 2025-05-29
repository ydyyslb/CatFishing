import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from causalnex.structure.notears import from_pandas
import networkx as nx
from causalnex.structure import StructureModel
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from hc.DAG import DAG
from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
def Notears_test(i2p,  **kwargs):
    # 读取数据
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    data = pd.read_csv(filename, dtype='category')
    exist_edges = kwargs.get('exist_edges', [])
    sm = StructureModel()
    # # 应用先验知识
    # exist_edges_str = [[i2p[edge[0]], i2p[edge[1]]] for edge in exist_edges]
    # for par, var in exist_edges_str:
    #     if par in sm.nodes and var in sm.nodes:
    #         sm.add_edge(par, var)
    # 非数值变量进行标签编码
    struct_data = data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        struct_data[col] = le.fit_transform(struct_data[col])
    # # 生成多项式特征
    # poly = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
    # X_poly = poly.fit_transform(struct_data)

    # 标准化处理
    X_np = struct_data.to_numpy()
    scaler = preprocessing.StandardScaler().fit(X_np)
    X_scaled = scaler.transform(X_np)
    # 处理禁止的边
    tabu_edges = kwargs.get('forb_edges', [])
    # 使用 i2p 将索引转换为字符串
    tabu_edges_str = [[i2p[edge[0]], i2p[edge[1]]] for edge in tabu_edges]
    # 将 X_scaled 转换为 pandas DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=struct_data.columns)

    # 使用 NOTEARS 算法学习结构
    sm = from_pandas(X_scaled_df, max_iter=300, w_threshold=0.8)

    # 转换为邻接矩阵，确保只包含 {0, 1}
    adj_matrix = nx.to_numpy_array(sm)
    adj_matrix = np.where(adj_matrix != 0, 1, 0)

    return adj_matrix