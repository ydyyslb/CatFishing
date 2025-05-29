import json
import math
import os
import re
import shutil
import time
import warnings
from typing import Optional

import chardet
import networkx as nx
import numpy as np
import pandas as pd
import requests
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.GESUtils import *
from causallearn.utils.TXT2GeneralGraph import mod_endpoint, to_endpoint
from retry import retry
from rich import print as rprint

# from simulator import simDAG

# 用于将一个 NumPy 数组转换为一个 GeneralGraph 对象。
# A: 一个二维 NumPy 数组，其中包含有向图的邻接矩阵。
def array2generalgraph(A: np.ndarray) -> GeneralGraph:
    # 获取数组 A 的行数和列数，分别存储在变量 n 和 m 中。
    n, m = A.shape
    g = GeneralGraph([])
    # 创建一个空字典 node_map，用于存储图中的节点和它们的索引。
    node_map = {}
    for i in range(n):
        node = 'X'+str(i+1)
        node_map[node] = GraphNode(node)
        g.add_node(node_map[node])
    B = A+A.T
    for i in range(n):
        for j in range(i+1, n):
            if B[i, j] == 1:
                node1 = 'X'+str(i+1)
                node2 = 'X'+str(j+1)
                edge = Edge(node_map[node1], node_map[node2],
                            Endpoint.CIRCLE, Endpoint.CIRCLE)
                if A[i, j] == 1:
                    mod_endpoint(edge, node_map[node2], Endpoint.ARROW)
                else:
                    mod_endpoint(edge, node_map[node2], Endpoint.TAIL)
                if A[j, i] == 1:
                    mod_endpoint(edge, node_map[node1], Endpoint.ARROW)
                else:
                    mod_endpoint(edge, node_map[node1], Endpoint.TAIL)
                g.add_edge(edge)
    return g

# 用于将一个字典转换为一个 GeneralGraph 对象。
# dict2generalgraph 函数用于将一个包含节点和父节点信息的字典转换为一个有向图对象。这个函数通过遍历字典中的每个键，
# 为每个键对应的节点创建一个 GraphNode 对象，并为每个键对应的父节点列表中的每个父节点创建一个从父节点到当前节点的有向边。
# A: 一个字典，其中包含有向图的节点和它们的父节点信息。
def dict2generalgraph(A: dict) -> GeneralGraph:
    #  创建一个新的 GeneralGraph 对象 g，并将其初始化为空图。
    g = GeneralGraph([])
    # 创建一个空字典 node_map，用于存储图中的节点和它们的索引。
    node_map = {}
    for key in A:
        #  在 node_map 字典中添加一个键值对，键是节点的名称，值是创建的 GraphNode 对象。
        node_map[key] = GraphNode(key)
        # 将 GraphNode 对象添加到图 g 中。
        g.add_node(node_map[key])
    for key in A:
        for pa in A[key]['par']:
            edge = Edge(node_map[pa], node_map[key],
                        Endpoint.TAIL, Endpoint.ARROW)
            g.add_edge(edge)
    return g

# directed_edge2array 函数用于将一个包含有向边信息的列表转换为一个 NumPy 数组，其中数组中的每个元素代表图中是否存在一条从相应的行索引到列索引的有向边。
def directed_edge2array(n: int, L: list) -> np.ndarray:
    A = np.zeros([n, n])
    for i in L:
        A[i[0], i[1]] = 1
    return A

# array2directed_edge 函数用于将一个 NumPy 数组转换为一个包含有向边信息的列表。通过找到数组中非零元素的索引，然后将这些索引配对成元组，最后将这些元组转换为列表。
def array2directed_edge(A: np.ndarray) -> list:
    a, b = np.where(A != 0)
    return list(zip(a, b))

# array2no_edge 函数用于将一个 NumPy 数组转换为一个包含无向边信息的列表。通过找到数组中所有元素为 0 的索引，然后将这些索引配对成元组，最后将这些元组转换为列表。
def array2no_edge(A: np.ndarray) -> list:
    a, b = np.where(A == 0)
    return list(zip(a, b))

# ShowGraph 函数用于将一个 GeneralGraph 对象转换为 Pydot 格式，然后将其渲染为 PNG 图像，并使用 matplotlib 模块在图形中显示该图像。这个函数适用于可视化图结构，使得图的节点和边以图形的形式直观地展示出来。
def ShowGraph(a: GeneralGraph):
    import io

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from causallearn.utils.GraphUtils import GraphUtils
    pyd = GraphUtils.to_pydot(a)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# 用于计算一个图模型（GeneralGraph）与数据集（ndarray）之间的评分。这个评分函数依赖于不同的评分函数和参数。
# score_func: 一个字符串，表示使用的评分函数，默认为 ‘local_score_BIC’。
# G: 一个 GeneralGraph 对象，表示图模型，默认为 None。
# maxP: 一个可选的浮点数，表示图模型中节点的最大父节点数量，默认为 None。
# parameters: 一个可选的字典，包含评分函数的参数，默认为 None。
def truth_score(X: ndarray, score_func: str = 'local_score_BIC', G: GeneralGraph = None, maxP: Optional[float] = None, parameters: Optional[Dict[str, Any]] = None):

    #  如果数据集的行数小于列数，即特征数量大于样本数量，
    if X.shape[0] < X.shape[1]:
        warnings.warn(
            "The number of features is much larger than the sample size!")

    # 将输入的 NumPy 数组转换为矩阵类型，以便后续使用。
    X = np.mat(X)
    # % k-fold negative cross validated likelihood based on regression in RKHS
    # 说明接下来的代码是使用 k 折交叉验证来计算基于 RKHS 的负对数似然评分。
    # 如果评分函数是 ‘local_score_CV_general’，
    if score_func == 'local_score_CV_general':
        if parameters is None:
            parameters = {'kfold': 10,  # 10 fold cross validation
                          'lambda': 0.01}  # regularization parameter
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters)

    # negative marginal likelihood based on regression in RKHS
    # 如果评分函数是 ‘local_score_marginal_general’
    elif score_func == 'local_score_marginal_general':
        parameters = {}
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters)

    # k-fold negative cross validated likelihood based on regression in RKHS
    # 如果评分函数是 ‘local_score_CV_multi’
    elif score_func == 'local_score_CV_multi':
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'kfold': 10, 'lambda': 0.01,
                          'dlabel': {}}  # regularization parameter
            for i in range(X.shape[1]):
                parameters['dlabel']['{}'.format(i)] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters)

    # negative marginal likelihood based on regression in RKHS
    elif score_func == 'local_score_marginal_multi':
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'dlabel': {}}
            for i in range(X.shape[1]):
                parameters['dlabel']['{}'.format(i)] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)

    # Greedy equivalence search with BIC score
    elif score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        parameters = {}
        parameters["lambda_value"] = 2
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)

    elif score_func == 'local_score_BDeu':  # Greedy equivalence search with BDeu score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None)

    else:
        raise Exception('Unknown function!')
    score_func = localScoreClass

    score = score_g(X, G, score_func, parameters)  # initialize the score

    return score

# dict2list 函数将一个包含图边信息的字典转换为一个包含特定格式字符串的列表。
# 这个列表中的每个字符串都包含了边的索引、边的权重、真实图中边的类型（无向、有向或不存在）、预测图中边的类型（无向、有向或不存在）以及分隔符 ‘;’。
# D: 一个字典，包含图的边信息。
# G: 一个 GeneralGraph 对象，表示真实图。
# P: 一个 GeneralGraph 对象，表示预测图。
def dict2list(D, G, P):
    result = []
    for edge in D:
        i, j = edge[1]
        # 检查真实图中位于 (i-1, j-1) 的元素是否为 -1，即是否存在无向边。
        if G.graph[i-1, j-1] == -1:
            flag1 = 'y'
        #  如果真实图中位于 (i-1, j-1) 的元素不为 -1
        else:
            # 检查真实图中位于 (i-1, j-1) 的元素是否为 1，即是否存在有向边。
            if G.graph[i-1, j-1] == 1:
                flag1 = 'r'
            else:
                flag1 = 'n'
        #   检查预测图中位于 (i-1, j-1) 的元素是否为 -1，即是否存在无向边。
        if P.graph[i-1, j-1] == -1:
            flag2 = 'y'
        # 如果预测图中位于 (i-1, j-1) 的元素不为 -1
        else:
            if P.graph[i-1, j-1] == 1:
                flag2 = 'r'
            else:
                flag2 = 'n'
        #  将边的索引、边的权重、真实图中边的类型、预测图中边的类型以及分隔符 ‘;’ 组合成一个字符串，并将其添加到 result 列表中。
        result.append(str((i, j, edge[0], flag1, flag2))+';')
    return result

# 用于将一个 NumPy 数组转换为一个字典，这个字典表示一个有向图的邻接信息。
def array2dict(A: np.ndarray, varnames):
    dag = {}
    n, m = A.shape
    for i in range(n):
        # 在字典 dag 中为当前行的节点名称创建一个键值对，值为一个空字典。
        dag[varnames[i]] = {}
        # 在当前节点的字典中创建一个键 ‘par’，其值为一个空列表
        dag[varnames[i]]['par'] = []
        # 在当前节点的字典中创建一个键 ‘nei’，其值为一个空列表。
        dag[varnames[i]]['nei'] = []
    for i in range(n):
        for j in range(m):
            # 如果 A 中位于 (i, j) 的元素为 1，表示 A 中位于 (j, i) 的元素为 1（有向边），或者两者都是 1（双向边）。
            if A[i, j] == 1:
                #  在节点 varnames[j] 的字典中，将节点 varnames[i] 添加到键 ‘par’ 的列表中，表示 varnames[i] 是 varnames[j] 的父节点。
                dag[varnames[j]]['par'].append(varnames[i])
    return dag

# 用于生成一个字符串，该字符串描述了图的前向状态。
# prior: 一个布尔值，表示是否包含先验信息，默认为 True。
# prior_type: 一个布尔值，表示是否包含先验类型的信息，默认为 True。
def generate_name(prior_state, prior=True, prior_type=True):
    st = ''
    for key in prior_state:
        # 如果当前键对应的先验信息为 True
        if prior_state[key][0] == True:
            if prior:
                st += key
            if prior_type:
                if prior_state[key][1] == True:
                    st += 'r'
                elif prior_state[key][1] == False:
                    st += 'p'
            st += ','
    #         移除字符串 st 末尾的分隔符 ‘,’。
    st = st.strip(',')
    # 如果 st 为空，则默认设置为 ‘n’
    if st == '':
        st = 'n'
    return st

# 用于解析实验结果文件，并将其转换为 Pandas DataFrame。
def parse_experiment_results_perform(file_name, column):
    with open(file_name, "r") as f:
        lines = f.readlines()
    res = pd.DataFrame(columns=column)
    for tmp in ["s", "r", "palim"]:
        if tmp in res.columns:
            res[tmp] = res[tmp].astype("int")
    tmp_record = {}
    for line in lines:
        if line[0] == '#':
            continue
        line = line.strip()
        if line.startswith("{"):
            line = line.replace("'", '"')
            line = line.replace("nan", 'null')
            tmp = json.loads(line)
            # add to the pd.dataframe res
            for key in tmp:
                # Add key as a new column name to res
                if key not in res.columns:
                    res[key] = None
                tmp_record[key] = float(
                    tmp[key]) if tmp[key] is not None else np.nan
            res = res.append(tmp_record, ignore_index=True)
            tmp_record = {}
        else:
            if line == '':
                continue
            v_list = line.split(" ")
            for k in v_list:
                key, value = k.split('=')
                try:
                    tmp_record[key] = eval(value)
                except:
                    tmp_record[key] = value
    return res


def parse_prior_results(file_name='exp/path_prior_evaluation.txt'):
    import json

    import pandas as pd
    with open(file_name, "r") as f:
        lines = f.readlines()
    res = pd.DataFrame(columns=["data"])
    props = ["data"]
    tmp_record = {}
    for line in lines:
        line = line.strip()
        if line.startswith("{"):
            line = line.replace("'", '"')
            line = line.replace("nan", 'null')
            tmp = json.loads(line)
            # add to the pd.dataframe res
            for key in tmp:
                # Add key as a new column name to res
                if key not in res.columns:
                    res[key] = None
                tmp_record[key] = float(
                    tmp[key]) if tmp[key] is not None else np.nan
            res = res.append(tmp_record, ignore_index=True)
            tmp_record = {}
        else:
            v_list = line.split(" ")
            for i, k in enumerate(props):
                tmp_record[k] = v_list[i]
    return res

# 这个函数的主要目的是将 CSV 格式的数据转换为 Txt 格式的数据，以便于后续的因果推断分析。
def ReconstructData(src_dir='data/csv', dst_dir='data/txt'):
    '''
    reconstruct the data:
    AAA   BBB   CCC
    True  High  Right
    into:
    0 1 2 3 (index)
    2 2 2 2 (arities)
    0 1 0 1 (data)
    '''
    # 读取 CSV 文件：使用 pd.read_csv 函数读取指定目录下的所有 CSV 文件。
    # 转换数据类型：将数据中的分类数据转换为整数编码。
    # 提取列名和基数：提取数据中的列名和每个列的唯一值数量。
    # 写入 Txt 文件：将转换后的数据、列名和基数写入指定目录的 Txt 文件中。
    for path in os.listdir(src_dir):
        data = pd.read_csv(f'{src_dir}/{path}', dtype='category')
        array_data = data.apply(lambda x: x.cat.codes).to_numpy(dtype=int)
        path = path.split('.')[0]
        arities = np.array(data.nunique())
        strtmp = ' '.join([str(i) for i in range(len(data.columns))])
        strtmp += '\n'
        strtmp += ' '.join([str(i) for i in arities])
        np.savetxt(f'{dst_dir}/{path}.txt', array_data,
                   fmt='%d', header=strtmp, comments='')

# 这个函数用于解析包含父节点和分数信息的 Txt 文件。具体步骤如下：
def parse_parents_score(file_path):
    score_dict = {}

    with open(file_path, 'r') as f:
        # read the first line to get the number of nodes (this value is not used in the example)
        node_number = int(f.readline().strip())

        while True:
            line = f.readline().strip()
            if not line:
                break

            # Extract information from the line
            parts = line.split()
            node_index = int(parts[0])
            num_parent_set = int(parts[1])

            # Create an empty list to hold the score and parent information for the current child node
            score_list = []

            # Loop over the number of parent sets and read each set of scores and parents
            for _ in range(num_parent_set):
                line = f.readline().strip()
                parent_parts = line.split()

                # Extract score and parent indices
                score = float(parent_parts[0])
                parent_num = int(parent_parts[1])
                parents = [str(x) for x in parent_parts[2: 2 + parent_num]]

                # Append to the list of scores and parents for this child node
                score_list.append((score, parents))

            # Save to the score_dict
            score_dict[str(node_index)] = score_list

    return score_dict

# 这个函数用于将提取的父节点和分数信息写入 Txt 文件。
def write_parents_score(score_dict, file_path):
    # 打开文件：使用 with open 语句打开文件，确保文件在使用后正确关闭。
    # 写入节点数量：首先写入文件中的节点数量。
    # 循环写入节点信息：对于每个节点，写入节点的索引、父节点集的数量，以及每个父节点集的分数和父节点列表。
    with open(file_path, 'w') as f:
        n = len(score_dict)
        f.write(f"{n}\n")
        for var in score_dict:
            f.write(f"{var} {len(score_dict[var])}\n")
            for score, parent_list in score_dict[var]:
                new_score = "{:.8f}".format(score)
                f.write(f"{new_score} {len(parent_list)} {' '.join(parent_list)}\n")
            
# 这个函数检查给定的 DAG 是否有从 source 到 dest 的有向路径。它使用广度优先搜索（BFS）来遍历 DAG。如果找到从 source 到 dest 的路径，则返回 True，否则返回 False。
# dag: 是一个 NumPy 数组，表示 DAG 的邻接矩阵。
# source: 源节点的索引。
# dest: 目标节点的索引。
def check_path(dag: np.array, source, dest):
    # Check if there is a directed path
    n = dag.shape[0]
    visited = np.zeros(n)
    queue = [source]
    while len(queue) > 0:
        v = queue.pop(0)
        for i in range(n):
            if dag[v][i] == 1 and visited[i] == 0:
                visited[i] = 1
                queue.append(i)
    return visited[dest] == 1

# 这个函数用于删除 DAG 中给定祖先和子节点对之间的所有路径。它使用 NetworkX 库来处理图结构，并删除所有可能的路径。
# dag: 是一个 NumPy 数组，表示 DAG 的邻接矩阵。
# ancs: 是一个包含祖先和子节点对的列表。
def delate_ancs(dag: np.array, ancs):
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    for anc, child in ancs:
        try:
            path = nx.shortest_path(G, source=anc, target=child)
        except:
            path = None
        while path != None:
            for i in range(len(path)-1):
                G.remove_edge(path[i], path[i+1])
            try:
                path = nx.shortest_path(G, source=anc, target=child)
            except:
                path = None
    # 返回一个 NumPy 数组，表示删除路径后的 DAG。
    return nx.to_numpy_array(G)

# 这个函数检查给定的 DAG 是否是无环的。它遍历所有节点，并检查是否存在从每个节点到自身的路径。如果存在，则返回 False，表示 DAG 不是无环的；否则返回 True。
def check_acyclic(dag: np.array):
    n = dag.shape[0]
    visited = np.zeros(n)
    for i in range(n):
        if visited[i] == 0:
            if check_path(dag, i, i):
                return False
    return True

# 这个函数尝试将一个有环的 DAG 转换为无环的 DAG。它遍历所有节点，并检查是否存在从每个节点到自身的路径。如果存在，它将移除环中的一个边。
def cyclic2acyclic(dag: np.array):
    # chekc if the dag is cyclic, if so, make it acyclic
    n = dag.shape[0]
    visited = np.zeros(n)
    for i in range(n):
        if visited[i] == 0:
            if check_path(dag, i, i):
                # find a cycle
                # find the edge in the cycle
                for j in range(n):
                    if dag[i][j] == 1 and check_path(dag, j, i):
                        dag[i][j] = 0
    # print('The algorithm for cycle removement is relatively simple, it can only ensure that the result is acyclic, but cannot ensure that the removed edges are minimal!')
    return dag

# 这个函数用于清除 DAG 中的循环。它首先创建一个与真实 DAG 形状相同的零矩阵，然后根据给定的顺序约束添加边。接着，它使用 cyclic2acyclic 函数尝试将 DAG 转换为无环的。
# true_dag: 真实 DAG 的邻接矩阵。
# order_constraints: 表示节点顺序的约束列表。
def clearcycle(true_dag, order_constraints):
    dag = np.zeros(true_dag.shape)
    for edge in order_constraints:
        dag[edge[0], edge[1]] = 1

    dag = cyclic2acyclic(dag)
    # print(check_acyclic(dag))
    edges = np.argwhere(dag == 1)
    edges = [(edge[0], edge[1]) for edge in edges]
    return edges

# 这个函数用于保存因果推断实验的结果。它首先打印实验的详细信息，然后将这些信息写入一个文件中。如果方法是 DP、Astar、ELSA 或 PGMINOBSx，它还会保存推断出的 DAG 到文件。
# m: 包含实验结果的指标对象。
# ev_dag: 推断出的 DAG。
# **kwargs: 包含实验参数的字典。
def save_result(m, ev_dag, **kwargs):

    print(f"d={kwargs['d']} s={kwargs['s']} r={kwargs['r']} conf={kwargs['conf']} palim={kwargs['palim']} prior={kwargs['prior']} prior_type={kwargs['prior_type']} pruning={kwargs['nopruning']} score={kwargs['score']}  prior_source={kwargs['prior_source']}\n{m.metrics}")
    nowtime = re.sub('\s+', '/', time.asctime(time.localtime()))
    with open(kwargs['output'], 'a') as f:
        f.write(f"d={kwargs['d']} s={kwargs['s']} r={kwargs['r']} conf={kwargs['conf']} palim={kwargs['palim']} prior={kwargs['prior']} prior_type={kwargs['prior_type']} pruning={kwargs['nopruning']} score={kwargs['score']}  prior_source={kwargs['prior_source']} finish_time={nowtime}\n{m.metrics}\n")

    if kwargs['method'] in ['DP', 'Astar', 'ELSA', 'PGMINOBSx']:
        np.savetxt(f"{kwargs['dag_path']}/{kwargs['method']}_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}_{kwargs['palim']}_{kwargs['prior']}_{kwargs['prior_type']}_{kwargs['nopruning']}_{kwargs['score']}_{kwargs['prior_source']}{kwargs['fix']}.txt", ev_dag, fmt="%d")
    elif kwargs['method'] == 'CaMML':
        np.savetxt(f"{kwargs['dag_path']}/{kwargs['method']}_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}_{kwargs['palim']}_{kwargs['prior']}_{kwargs['prior_type']}_{kwargs['prior_source']}{kwargs['fix']}.txt", ev_dag, fmt="%d")
    else:
        np.savetxt(f"{kwargs['dag_path']}/{kwargs['method']}_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}_{kwargs['conf']}_{kwargs['prior']}_{kwargs['prior_type']}_{kwargs['prior_source']}{kwargs['fix']}.txt", ev_dag, fmt="%d")

# 这个函数用于计算因果推断实验的指标，并将其保存到 CSV 文件中。它首先解析输入文件，计算指标，然后根据 cat_column 和 merge_column 进行分组和聚合。
# input: 包含实验结果的输入文件路径。
# output: 包含实验结果统计的输出文件路径。
# cat_column: 用于分组和聚合的列名列表。
# merge_column: 用于聚合的列名列表。
# mean: 如果为 True，则计算每个组的平均值；否则计算每个组的值。
def noprior_dag_metric(input="exp/CaMML_noprior.txt", output="exp/CaMML_noprior_statistics.csv", cat_column=["prior", "prior_type", "palim", "d", "s"], merge_column=["r"], mean=True):
    res = parse_experiment_results_perform(
        input, column=cat_column+merge_column)
    # res=res.dropna()
    res['precision'] = res['precision'].fillna(0)
    res['F1'] = res['F1'].fillna(0)

    warnings.filterwarnings("ignore")
    if mean:
        res = res.groupby(cat_column).mean()
    else:
        res.sort_values(cat_column, inplace=True)
        res.reset_index(drop=True, inplace=True)
    res.drop(labels=['fdr', 'tpr', 'fpr', 'nnz', 'r', 'gscore', 'delp_fdr', 'delp_tpr',
             'delp_fpr', 'delp_nnz', 'delp_gscore'], axis=1).to_csv(output, float_format="%.2f")

# 创建指定路径的目录，如果目录已存在，则不进行任何操作。
def mkdir(path):
    """
    make directory, if the directory exists, do nothing
    """
    # 使用os.path.exists函数检查目录是否存在。
    # 如果目录不存在，则使用os.makedirs函数创建目录。
    # 使用rprint函数打印创建目录的提示信息。
    if not os.path.exists(path):
        os.makedirs(path)
        rprint(f"📂 Created folder [italic blue]{path}[/italic blue].")

# 创建指定路径的目录，如果目录已存在，则先删除目录，然后创建新的目录。
def mkdir_rm(path):
    """
    make directory, if the directory exists, remove it and create a new one
    """
    # 使用os.path.exists函数检查目录是否存在。
    # 如果目录存在，则使用shutil.rmtree函数删除目录。
    # 使用rprint函数打印删除目录的提示信息。
    # 然后调用mkdir函数创建目录。
    if os.path.exists(path):
        shutil.rmtree(path)
        rprint(f"🗑️  Removed folder [italic blue]{path}[/italic blue].")
    mkdir(path)


# 自动清理指定文件夹中的文件，保留最新的max_num个文件。
def auto_clean_folder(folder_path, max_num=50):
    # folder_path: 字符串，表示要清理的文件夹的路径。
    # max_num: 整数，表示保留的文件数量，默认为50。
    """
    an automatic clean function for folder, keep the latest max_num files
    """
    # 使用os.listdir函数获取文件夹中的文件列表。
    # 按修改时间对文件列表进行排序，以确保最新的文件排在前面。
    # 计算需要删除的文件数量。
    # 遍历文件列表，删除旧文件。
    # 使用rprint函数打印清理文件的提示信息。
    folder_num = len(os.listdir(folder_path))
    if folder_num > max_num:
        file_list = os.listdir(folder_path)
        file_list.sort(key=lambda fn: os.path.getmtime(
            os.path.join(folder_path, fn)))
        num_to_remove = len(file_list) - max_num
        for i in range(num_to_remove):
            file_path = os.path.join(folder_path, file_list[i])
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.chmod(file_path, 0o777)
                shutil.rmtree(file_path)
            rprint(f"🗑️  Removed file [italic blue]{file_path}[/italic blue].")
        rprint(
            f"♻️  [bold yellow]Auto clean[/bold yellow] {num_to_remove} files in [italic blue]{folder_path}[/italic blue].")

# 将内容写入指定路径的文本文件，如果文件已存在，则追加内容；如果文件不存在，则创建文件并写入内容。
# txt_path: 字符串，表示文本文件的路径。
# content: 字符串，表示要写入文件的内容。
# mode: 字符串，表示写入模式，默认为"w"（覆盖写入），可选为"a"（追加写入）。
def write_txt(txt_path, content,mode="w"):
    """
    write content to txt file
    """
    with open(txt_path, mode, encoding='utf-8') as f:
        f.write(content)
    if mode=="w":
        rprint(
            f"📝 Write [bold yellow]txt[/bold yellow] file to [italic blue]{txt_path}[/italic blue].")
    else:
        rprint(
            f"📝 Append [bold yellow]txt[/bold yellow] file to [italic blue]{txt_path}[/italic blue].")


# 这个函数用于自动检测文件的字符编码。它使用 chardet 库来检测文件内容的编码，并将检测到的编码写入控制台。
# 参数：
# file_path: 文件的路径。
# 返回值：
# 返回文件的字符编码。
def auto_detect_encoding(file_path):
    """
    detect encoding of file automatically
    """
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    rprint(
        f"🔍 Detected encoding: [bold yellow]{encoding}[/bold yellow] of [italic blue]{file_path}[/italic blue].")
    return encoding


# 这个函数用于读取文本文件。如果指定 encoding 为 'auto'，它会自动检测文件的编码。
# txt_path: 文本文件的路径。
# encoding: 文件的字符编码。
def read_txt(txt_path, encoding='utf-8'):
    """
    read txt file
    use utf-8 as default encoding, if you want to auto detect encoding, set encoding='auto'
    """
    if encoding == 'auto':
        encoding = auto_detect_encoding(txt_path)
    with open(txt_path, "r", encoding=encoding) as f:
        content = f.read()
    rprint(
        f"📖 Read [bold yellow]txt[/bold yellow] file from [italic blue]{txt_path}[/italic blue].")
    # 返回文本文件的内容。
    return content

# 这个函数用于将内容写入 JSON 文件。它可以接受字典、列表或其他可序列化为 JSON 的对象。
# content: 需要写入的 JSON 序列化对象。
# json_path: JSON 文件的路径。
# encoding: JSON 文件的编码。
# indent: JSON 文件的缩进。
def write_json(content, json_path, encoding='utf-8', indent=4):
    """
    Write content to json file.

    Args:
        content: dict, list, or other json serializable object.
        json_path: str, path to json file.
        encoding: str, encoding of json file.
        indent: int, indent of json file.
    """
    try:
        # 确保包含文件的目录存在
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding=encoding) as f:
            json.dump(content, f, ensure_ascii=False, indent=indent)
        rprint(
            f"📝 Write [bold yellow]json[/bold yellow] file to [italic blue]{json_path}[/italic blue].")
    except IOError as e:
        # 处理IO错误，例如权限问题或其他I/O相关错误
        rprint(f"An IOError occurred: {e}")
    except Exception as e:
        # 处理其他可能的异常
        rprint(f"An unexpected error occurred: {e}")

# 从指定的JSON文件路径读取文件内容。
# json_path: 字符串，表示JSON文件的路径。
# encoding: 字符串，表示文件编码，默认为’utf-8’，如果指定为’auto’，则自动检测编码。
# quiet: 布尔值，如果为False，则打印成功读取的提示信息，默认为False。
def read_json(json_path, encoding='utf-8',quiet=False):
    """
    read json file
    """
    # 如果指定encoding为’auto’，则使用auto_detect_encoding函数自动检测文件编码。
    # 使用open函数以读取模式打开文件，并指定正确的编码。
    # 使用json.load函数读取文件内容。
    # 如果quiet为False，则使用rprint函数打印成功读取的提示信息。
    # 返回读取的JSON内容。
    if encoding == 'auto':
        encoding = auto_detect_encoding(json_path)
    with open(json_path, "r", encoding=encoding) as f:
        content = json.load(f)
    if not quiet:
        rprint(
            f"📖 Read [bold yellow]json[/bold yellow] file from [italic blue]{json_path}[/italic blue].")
    #     返回值：读取的JSON内容，以Python字典或列表的形式返回。
    return content

# 读取JSON文件，更新文件内容，并写回文件。
# add_content: 字典或列表，表示要添加到JSON文件的内容。
# json_path: 字符串，表示JSON文件的路径。
# encoding: 字符串，表示文件编码，默认为’utf-8’，如果指定为’auto’，则自动检测编码。
def update_json(add_content, json_path, encoding='utf-8'):
    """
    update json file
    """
    # 如果指定encoding为’auto’，则使用auto_detect_encoding函数自动检测文件编码。
    # 使用open函数以读取模式打开文件，并指定正确的编码。
    # 使用json.load函数读取文件内容。
    # 将add_content添加到content字典或列表中。
    # 使用open函数以写入模式打开文件，并指定正确的编码。
    # 使用json.dump函数将更新后的内容写回文件，并指定确保非ASCII字符的输出、缩进等参数。
    # 使用rprint函数打印成功更新的提示信息。
    if encoding == 'auto':
        encoding = auto_detect_encoding(json_path)
    with open(json_path, "r", encoding=encoding) as f:
        content = json.load(f)
    content.update(add_content)
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    rprint(
        f"🔄 Update [bold yellow]json[/bold yellow] file to [italic blue]{json_path}[/italic blue].")

# 读取JSON文件，追加内容到文件，并写回文件
def append_json(add_content, json_path, encoding='utf-8'):
    """
    append json file
    """
    if encoding == 'auto':
        encoding = auto_detect_encoding(json_path)
    with open(json_path, "r", encoding=encoding) as f:
        content = json.load(f)
    content.append(add_content)
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    rprint(
        f"🔄 Update [bold yellow]json[/bold yellow] file to [italic blue]{json_path}[/italic blue].")

# 这个装饰器用于过滤警告。
# 它首先调用 warnings.filterwarnings("ignore") 来禁用警告，
# 然后执行原始函数 func，
# 最后调用 warnings.filterwarnings("default") 来恢复默认警告设置。

# func: 需要装饰的函数。
def warning_filter(func):
    """
    a decorator to filter warnings
    """
    def inner(*args, **kwargs):
        warnings.filterwarnings("ignore")
        result = func(*args, **kwargs)
        warnings.filterwarnings("default")
        return result
    # 返回值：
    # 返回原始函数 func 的执行结果。
    return inner

# 这个装饰器用于计算函数的执行时间。它首先记录函数开始执行的时间，然后执行原始函数 func，最后记录函数结束执行的时间，并计算时间差。
def timer(func):
    """
    a decorator to calculate the time cost of a function
    """
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        rprint(
            f"⏱️ Function [italic blue]{func.__name__}[/italic blue] cost [bold yellow]{end-start:.4f}[/bold yellow] seconds.")
        return result
    return inner

# 这个函数用于对列表进行排序。
# 它接受一个包含元组的列表，每个元组包含两个元素：第一个元素是一个键值对，第二个元素是一个值。函数将列表中的元组按照键进行排序，并将排序后的结果存储在字典中。
def sort(list):
    result = {}
    for tuple in list:
        if tuple[0][1] not in result.keys():
            result[tuple[0][1]] = {}
            result[tuple[0][1]][tuple[0][0]] = tuple[1]
        else:
            result[tuple[0][1]][tuple[0][0]] = tuple[1]
    # 返回一个字典，其中键是排序后的键值对，值是相应的值。
    return result


# 这段代码定义了一个名为 soft_constraint 的类，用于表示软约束。
# 软约束是指在因果推断中，某些边可能是强制的（obligatory），而其他边则可能被禁止（forbidden）。
# 这些约束通过 lambda 函数来表示，其中 lambda[0] 表示强制边的权重，而 lambda[1] 表示禁止边的权重。
class soft_constraint:
    # obligatory: 强制边的列表，每个边由两个元素组成，分别表示父节点和子节点。
    # forbidden: 禁止边的列表，每个边由两个元素组成，分别表示父节点和子节点。
    # lamdba: 权重列表，包含强制边权重和禁止边权重。
    def __init__(self, obligatory, forbidden, lamdba):
        # 3 parameters: obligatory edges, forbidden edges, lambda
        # 属性:
        # obligatory: 存储强制边的字典，其中键是子节点，值是父节点的权重。
        # forbidden: 存储禁止边的字典，其中键是子节点，值是父节点的权重。
        self.obligatory = sort([(x, y) for x, y in zip(obligatory, lamdba[0])])
        self.forbidden = sort([(x, y) for x, y in zip(forbidden, lamdba[1])])

    # 计算给定变量 var 在其父节点集合 parent 下的先验得分。这个得分反映了软约束对推断的影响，其中软约束包括强制边（obligatory edges）和禁止边（forbidden edges）。
    # var: 当前变量。
    # parent: 当前变量的父节点集合。
    def calculate(self, var, parent):
        prior_score = 0
        if var in self.obligatory.keys():
            for p in self.obligatory[var].keys():
                if p in parent:
                    prior_score += math.log(self.obligatory[var][p])
                else:
                    prior_score += math.log(1-self.obligatory[var][p])
        if var in self.forbidden.keys():
            for p in self.forbidden[var].keys():
                if p in parent:
                    prior_score += math.log(1-self.forbidden[var][p])
                else:
                    prior_score += math.log(self.forbidden[var][p])
        # 返回 prior_score，即根据当前变量和父节点计算出的先验得分。
        return prior_score

class ColorP:
    def __init__(self) -> None:
        pass

    # 功能：将给定的边添加颜色，使其呈现紫色。
    # 参数：
    # edge: 列表，包含两个元素，分别表示边的起点和终点。
    @staticmethod
    def edge(edge):
        start, end = edge
        # # 返回值：字符串，表示带有颜色的边。
        return f"[purple]{start}[/purple]->[purple]{end}[/purple]"
    # 根据答案和真实答案的匹配情况，添加不同的颜色和样式。
    # answer: 字符串，表示用户的答案。
    # true_ans: 字符串，表示真实的答案。
    # 字符串，表示带有颜色和样式的答案。
    @staticmethod
    def answer(answer, true_ans):
        if answer == "D":  # answer is uncertain
            return f"(Ans: [yellow]{answer}[/yellow] / [green]{true_ans}[/green])"
        elif answer == true_ans:  # answer is correct
            if answer in ["B", "C"]:  # the correct answer makes effect
                return f"(Ans: [bold green]{answer}[/bold green] / [green]{true_ans}[/green])"
            else:  # the correct answer does not make effect
                return f"(Ans: [green]{answer}[/green] / [green]{true_ans}[/green])"
        elif answer == "A":  # answer is wrong, but does not make effect
            return f"(Ans: [yellow]{answer}[/yellow] / [green]{true_ans}[/green])"
        else:  # answer is wrong, and makes effect
            return f"(Ans: [bold red]{answer}[/bold red] / [green]{true_ans}[/green])"
    # 将给定的真实答案添加颜色，使其呈现黄色。
    # GT: 字符串，表示真实答案
    @staticmethod
    def GT(GT):
        # 字符串，表示带有颜色的真实答案。
        return f"(TrueAns: [yellow]{GT}[/yellow])"

    # 功能：将给定的模型添加颜色和样式，使其呈现黄色并加粗。
    # 参数：
    # model: 字符串，表示模型名称或描述。
    # 返回值：字符串，表示带有颜色和样式的模型。
    @staticmethod
    def model(model):
        return f"[bold yellow]{model}[/bold yellow]"

    # 功能：将给定的路径添加颜色和样式，使其呈现蓝色并倾斜。
    # 参数：
    # path: 字符串，表示路径。
    # 返回值：字符串，表示带有颜色和样式的路径。
    @staticmethod
    def path(path):
        return f"[italic blue]{path}[/italic blue]"

    # 功能：将给定的警告内容添加颜色，使其呈现红色。
    # 参数：
    # content: 字符串，表示警告信息。
    # 返回值：字符串，表示带有颜色的警告信息。
    @staticmethod
    def warning(content):
        return f"[red]{content}[/red]"

if __name__ == "__main__":
    score_dict = parse_parents_score("data/score/bdeu/asia_1000_1.txt")
    write_parents_score(score_dict,"test_score.tmp")