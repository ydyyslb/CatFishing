from hc.accessory import local_score, local_score_from_storage
from hc.DAG import DAG, check_cycle
from utils import parse_parents_score
import numpy as np
import itertools


def gsp(data, arities, varnames, pc=None, score='default'):
    '''
    GSP (Greedy Sparsest Permutation) causal discovery algorithm
    :param data: the training data used for learn BN (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param pc: the candidate parents and children set for each variable
    :param score: score function, including:
                   bic (Bayesian Information Criterion for discrete variable)
                   bic_g (Bayesian Information Criterion for continuous variable)

    :return: the learned BN (bnlearn format)
    '''
    if score == 'default':
        score = 'bic_g' if arities is None else 'bic'

    n = len(varnames)
    variables = list(range(n))

    # 初始化DAG和缓存
    dag = {}
    cache = {}
    for var in varnames:
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = []
        cache[var] = {}
        cache[var][tuple([])] = local_score(
            data, arities, [varnames.index(var)], score)

    # 初始化局部得分
    local_scores = {}
    for i in range(n):
        local_scores[i] = {}
        # 减少父节点组合的考虑数量，只考虑最多4个父节点
        max_parents = min(n - 1, 4)
        for parents in itertools.chain.from_iterable(itertools.combinations(
                [j for j in variables if j != i], r) for r in range(max_parents + 1)):
            cols = [i] + list(parents)
            local_scores[i][parents] = local_score(data, arities, cols, score)

    # 找到最优排列
    best_perm = find_best_permutation_improved(n, local_scores)

    # 根据最优排列构建DAG，使用贪心算法添加边
    return build_dag_from_perm(best_perm, varnames, local_scores, dag)


def find_best_permutation_improved(n, local_scores):
    '''
    Find the best causal ordering permutation using a greedy approach with improved efficiency
    :param n: number of variables
    :param local_scores: dictionary of local scores for each variable and its parent set
    :return: the best permutation
    '''
    # 初始化
    remaining = set(range(n))
    ordering = []

    # 贪婪地构建最优排列
    while remaining:
        best_score = float('-inf')
        best_var = None

        # 对于每个剩余变量，计算将其作为下一个变量的得分
        for var in remaining:
            # 计算可能的父节点（当前已排序的变量）
            if len(ordering) > 4:  # 如果已排序的变量超过4个，只考虑最优的4个
                # 为了提高效率，随机选择4个已排序的变量作为可能的父节点
                possible_parents = tuple(sorted(np.random.choice(ordering, 4, replace=False)))
            else:
                possible_parents = tuple(sorted([v for v in ordering]))

            # 获取当前变量给定可能父节点的得分
            if possible_parents in local_scores[var]:
                score = local_scores[var][possible_parents]
                if score > best_score:
                    best_score = score
                    best_var = var

        # 如果找不到最优变量（可能因为父节点集太大），则随机选择一个
        if best_var is None:
            best_var = next(iter(remaining))

        # 将最优变量添加到排列中
        ordering.append(best_var)
        remaining.remove(best_var)

    return ordering


def build_dag_from_perm(perm, varnames, local_scores, dag):
    '''
    Build a DAG from a permutation using a greedy approach similar to HC
    :param perm: variable permutation
    :param varnames: variable names
    :param local_scores: dictionary of local scores
    :param dag: initial DAG structure
    :return: the learned BN
    '''
    n = len(perm)

    # 类似HC算法，使用贪婪迭代方式构建DAG
    diff = 1
    while diff > 0:
        diff = 0
        edge_candidate = []

        # 遍历变量，尝试添加新边
        for i, var_idx in enumerate(perm):
            var = varnames[var_idx]

            # 考虑在排列中位于当前变量之前的变量作为潜在父节点
            for j in range(i):
                parent_idx = perm[j]
                parent = varnames[parent_idx]

                # 只有当parent不是var的父节点时尝试添加边
                if parent not in dag[var]['par']:
                    # 检查是否会形成环
                    if not check_cycle(parent, var, dag):
                        # 计算添加边后的得分变化
                        current_par_tuple = tuple(sorted([varnames.index(p) for p in dag[var]['par']]))
                        new_par_tuple = tuple(sorted(list(current_par_tuple) + [parent_idx]))

                        # 获取当前得分和添加边后的得分
                        current_score = local_scores[var_idx][current_par_tuple] if current_par_tuple in local_scores[
                            var_idx] else 0
                        if new_par_tuple in local_scores[var_idx]:
                            new_score = local_scores[var_idx][new_par_tuple]
                            score_diff = new_score - current_score

                            # 如果得分提高，记录这个边候选
                            if score_diff > diff:
                                diff = score_diff
                                edge_candidate = [parent, var, 'a']  # 'a' 表示添加边

        # 遍历变量，尝试删除现有边
        for var in varnames:
            for parent in list(dag[var]['par']):
                # 尝试删除边
                current_par_tuple = tuple(sorted([varnames.index(p) for p in dag[var]['par']]))
                new_par_tuple = tuple(sorted([varnames.index(p) for p in dag[var]['par'] if p != parent]))

                # 获取当前得分和删除边后的得分
                current_score = local_scores[varnames.index(var)][current_par_tuple] if current_par_tuple in \
                                                                                        local_scores[
                                                                                            varnames.index(var)] else 0
                if new_par_tuple in local_scores[varnames.index(var)]:
                    new_score = local_scores[varnames.index(var)][new_par_tuple]
                    score_diff = new_score - current_score

                    # 如果得分提高，记录这个边候选
                    if score_diff > diff:
                        diff = score_diff
                        edge_candidate = [parent, var, 'd']  # 'd' 表示删除边

        # 如果找到了得分提高的边操作，执行它
        if edge_candidate:
            if edge_candidate[2] == 'a':  # 添加边
                dag[edge_candidate[1]]['par'].append(edge_candidate[0])
                dag[edge_candidate[1]]['par'].sort()
            elif edge_candidate[2] == 'd':  # 删除边
                dag[edge_candidate[1]]['par'].remove(edge_candidate[0])

    return dag


def gsp_prior(D: DAG, score_filepath=None):
    '''
    GSP with prior knowledge, improved version
    :param D: DAG class containing data, variables and prior knowledge
    :param score_filepath: file path for precomputed scores
    :return: the learned BN (bnlearn format)
    '''
    score_dict = {}
    if score_filepath is not None:
        score_dict = parse_parents_score(score_filepath)

    if D.score == 'default':
        D.score = 'bic_g' if D.arities is None else 'bic'

    n = len(D.varnames)
    variables = list(range(n))

    # 初始化DAG和缓存
    dag = {}
    cache = {}
    for var in D.varnames:
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = []
        cache[var] = {}
        if score_filepath is not None:
            cache[var][tuple([])] = local_score_from_storage(
                [D.varnames.index(var)], score_dict)
        else:
            cache[var][tuple([])] = local_score(
                D.data, D.arities, [D.varnames.index(var)], D.score)

    # 应用先验知识到初始DAG
    for var in D.varnames:
        dag[var]['par'] = sorted(dag[var]['par'] + D.a_prior[var]['par'])

    # 初始化局部得分，减少父节点组合的考虑数量
    local_scores = {}
    for i in range(n):
        local_scores[i] = {}
        max_parents = min(n - 1, 4)  # 最多考虑4个父节点以提升性能
        for parents in itertools.chain.from_iterable(itertools.combinations(
                [j for j in variables if j != i], r) for r in range(max_parents + 1)):
            cols = [i] + list(parents)
            if score_filepath is not None:
                local_scores[i][parents] = local_score_from_storage(cols, score_dict)
            else:
                local_scores[i][parents] = local_score(D.data, D.arities, cols, D.score)

    # 获取初始排列，这个排列考虑了先验知识约束
    initial_perm = find_best_constrained_permutation_improved(n, local_scores, D)

    # 使用HC风格的迭代改进方法构建和优化DAG
    dag = build_dag_with_prior(initial_perm, D, local_scores, dag, score_filepath)

    D.dag = dag
    return dag


def find_best_constrained_permutation_improved(n, local_scores, D):
    '''
    Find the best causal ordering permutation with improved efficiency
    :param n: number of variables
    :param local_scores: dictionary of local scores
    :param D: DAG class containing prior knowledge
    :return: the best permutation
    '''
    # 初始化
    remaining = set(range(n))
    ordering = []

    # 创建必需的顺序约束
    must_come_before = {i: set() for i in range(n)}

    # 根据先验知识添加约束
    for i, var_i in enumerate(D.varnames):
        for j, var_j in enumerate(D.varnames):
            if var_i != var_j:
                # 如果var_i是var_j的先验父节点，那么var_i必须在var_j之前
                if var_i in D.a_prior[var_j]['par']:
                    must_come_before[j].add(i)

    # 贪婪地构建最优排列，同时考虑约束
    while remaining:
        best_score = float('-inf')
        best_var = None

        # 找出所有可以添加的变量（没有未处理的约束）
        available = [var for var in remaining if all(
            constraint not in remaining for constraint in must_come_before[var])]

        # 如果没有可用变量，则存在循环依赖，随机选择一个变量
        if not available:
            available = list(remaining)

        # 在可用变量中寻找得分最高的，限制父节点数量以提高效率
        for var in available:
            # 如果已排序的变量超过4个，只考虑最优的4个作为可能的父节点
            if len(ordering) > 4:
                possible_parents = tuple(sorted(np.random.choice(ordering, 4, replace=False)))
            else:
                possible_parents = tuple(sorted([v for v in ordering]))

            # 计算当前变量给定可能父节点的得分
            if possible_parents in local_scores[var]:
                score = local_scores[var][possible_parents]
                if score > best_score:
                    best_score = score
                    best_var = var

        # 如果找不到最优变量，则随机选择一个
        if best_var is None and available:
            best_var = available[0]

        # 将最优变量添加到排列中
        if best_var is not None:
            ordering.append(best_var)
            remaining.remove(best_var)

    return ordering


def build_dag_with_prior(perm, D, local_scores, dag, score_filepath):
    '''
    Build and optimize DAG with prior knowledge using HC-like approach
    :param perm: initial variable permutation
    :param D: DAG class containing prior knowledge
    :param local_scores: dictionary of local scores
    :param dag: initial DAG structure
    :param score_filepath: file path for precomputed scores
    :return: optimized DAG
    '''
    # 使用类似于HC的迭代优化策略
    diff = 1
    while diff > 0:
        diff = 0
        edge_candidate = []

        # 遍历所有变量对，尝试添加边
        for i, var_i in enumerate(D.varnames):
            for var_j in D.varnames:
                if var_i != var_j and var_j not in D.d_prior[var_i]['par'] and var_i not in dag[var_j]['par']:
                    # 检查是否会形成环
                    if not check_cycle(var_i, var_j, dag):
                        # 计算添加边后的得分变化
                        current_par = dag[var_j]['par']
                        current_par_tuple = tuple(sorted([D.varnames.index(p) for p in current_par]))
                        new_par_tuple = tuple(sorted(list(current_par_tuple) + [D.varnames.index(var_i)]))

                        # 获取得分
                        var_idx = D.varnames.index(var_j)
                        if current_par_tuple in local_scores[var_idx] and new_par_tuple in local_scores[var_idx]:
                            current_score = local_scores[var_idx][current_par_tuple]
                            new_score = local_scores[var_idx][new_par_tuple]
                            score_diff = new_score - current_score

                            # 如果得分提高，记录这个边候选
                            if score_diff > diff:
                                diff = score_diff
                                edge_candidate = [var_i, var_j, 'a']  # 'a' 表示添加边

        # 尝试删除非先验必需的边
        for var in D.varnames:
            for par in dag[var]['par']:
                if par not in D.a_prior[var]['par']:  # 不尝试删除先验必需的边
                    # 计算删除边后的得分变化
                    current_par = dag[var]['par']
                    current_par_tuple = tuple(sorted([D.varnames.index(p) for p in current_par]))
                    new_par_tuple = tuple(sorted([D.varnames.index(p) for p in current_par if p != par]))

                    # 获取得分
                    var_idx = D.varnames.index(var)
                    if current_par_tuple in local_scores[var_idx] and new_par_tuple in local_scores[var_idx]:
                        current_score = local_scores[var_idx][current_par_tuple]
                        new_score = local_scores[var_idx][new_par_tuple]
                        score_diff = new_score - current_score

                        # 如果得分提高，记录这个边候选
                        if score_diff > diff:
                            diff = score_diff
                            edge_candidate = [par, var, 'd']  # 'd' 表示删除边

        # 如果找到了得分提高的边操作，执行它
        if edge_candidate:
            if edge_candidate[2] == 'a':  # 添加边
                dag[edge_candidate[1]]['par'].append(edge_candidate[0])
                dag[edge_candidate[1]]['par'].sort()
            elif edge_candidate[2] == 'd':  # 删除边
                dag[edge_candidate[1]]['par'].remove(edge_candidate[0])

    return dag