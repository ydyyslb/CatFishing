import copy

import numpy as np
import pandas as pd

# 负责有向无环图（DAG）的评估。
class MetricsDAG:
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP / (TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP / (TP + FP)
    recall: TP / (TP + FN)
    F1: 2*(recall*precision) / (recall + precision)
    gscore: max(0, (TP - FP)) / (TP + FN), A score ranges from 0 to 1

    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    计算B_est的各种准确性指标。
    真阳性 （TP）：以正确方向估计的边缘。
    true nagative（TN）：既不在估计图中也不在真实图中的边缘。
    误报 （FP）：在估计图中但不在真实图中的边缘。
    假阴性 （FN）：不在估计图中但位于真实图中的边缘。
    反向 = 用反向方向估计的边缘。

fdr：（反向 + FP）/（TP + FP）
    总吨位：TP / （TP + FN）
    fpr：（反向 + FP）/（TN + FP）
    SHD：无定向额外 + 无定向缺失 + 反向
    新西兰：TP + FP
    精度：TP / （TP + FP）
    召回：TP / （TP + FN）
    F1：2*（召回率*精度）/（召回率+精度）
    gscore： max（0， （TP - FP）） / （TP + FN）， A 分数范围从 0 到 1

参数
    ----------
    B_est： np.ndarray
        [d， d] 估计值， {0， 1， -1}， -1 是 CPDAG 中的无向边。
    B_true：np.ndarray
        [d， d] 真值图， {0， 1}.
    """


    # 通过 __init__ 方法，它可以接收估计图和真实图，并计算它们的准确度指标。
    def __init__(self, B_est, B_true):
        # B_est: 估计的 DAG 邻接矩阵，通常是通过某种算法从数据中学习得到的。
        # B_true: 真实的 DAG 邻接矩阵，即在真实数据生成过程中定义的图结构。
        # 检查 B_est 是否是一个 NumPy 数组。
        if not isinstance(B_est, np.ndarray):
            # 如果 B_est 不是 NumPy 数组，则抛出一个 TypeError。
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        #  将 B_est 参数的值复制到类的实例属性 self.B_est 中。
        self.B_est = copy.deepcopy(B_est)
        # 将 B_true 参数的值复制到类的实例属性 self.B_true 中。
        self.B_true = copy.deepcopy(B_true)
        # 该方法计算估计图和真实图之间的准确度指标，并将结果存储在类的实例属性 self.metrics 中。
        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)


    # @staticmethod: 这是一个装饰器，表明 _count_accuracy 是一个静态方法，它不依赖于类的实例状态，可以直接通过类名调用。
    @staticmethod
    # B_est: 估计的 DAG 邻接矩阵，可以是 {0, 1} 或 {0, 1, -1}。
    # B_true: 真实的 DAG 邻接矩阵，通常是 {0, 1}。
    # decimal_num: 结果的小数位数，默认为 4。
    def _count_accuracy(B_est, B_true, decimal_num=4):

        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        decimal_num: int
            Result decimal numbers.

        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1

                参数
        ----------
        B_est： np.ndarray
            [d， d] 估计值， {0， 1， -1}， -1 是 CPDAG 中的无向边。
        B_true：np.ndarray
            [d， d] 真值图， {0， 1}.
        decimal_num： int
            结果：十进制数。

        返回
        ------
        指标：dict
            FDR：浮点数
                （反向 + FP） / （TP + FP）
            TPR：浮点
                TP/（TP + FN）
            FPR：浮动
                （反向 + FP） / （TN + FP）
            SHD：整数
                无向额外 + 无向缺失 + 反向
            新西兰： int
                TP + FP系列
            精度：浮点
                TP/（TP + FP）
            召回：浮动
                TP/（TP + FN）
            F1：浮点
                2*（召回率*精度）/（召回率+精度）
            gscore：浮点数
                max（0， （TP-FP））/（TP+FN），得分范围为 0 到 1
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            # 检查 B_est 是否包含 -1，即是否为 CPDAG（条件依存有向无环图）
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0
        #  检查 B_est 是否包含 -1，即是否为 CPDAG（条件依存有向无环图）
        if (B_est == -1).any():  # cpdag
            #  确保 B_est 只包含 {0, 1, -1} 中的值。
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            # 确保每个无向边只出现一次。
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        #     如果 B_est 不是 CPDAG，
        else:  # dag
            #  确保 B_est 只包含 {0, 1} 中的值。
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
        #     获取真实图的大小 d。
        d = B_true.shape[0]

        # linear index of nonzeros
        # 找到 B_est 中的无向边。
        pred_und = np.flatnonzero(B_est == -1)
        # 找到 B_est 中的有向边。
        pred = np.flatnonzero(B_est == 1)
        # 找到真实图中的有向边。
        cond = np.flatnonzero(B_true)
        # 找到真实图中的无向边。
        cond_reversed = np.flatnonzero(B_true.T)
        # 合并真实图中的有向边和无向边。
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        # 找到估计图和真实图中的共同有向边
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        # 找到估计图和真实图中的共同无向边。
        true_pos_und = np.intersect1d(
            pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        # 找到估计图中的错误有向边。
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        # 找到估计图中的错误无向边。
        false_pos_und = np.setdiff1d(
            pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        #  找到估计图中的额外有向边。
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        #  找到估计图中的反转有向边。
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        # 计算 FDR（假发现率）。
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(
            cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {'extra': len(extra_lower), 'missing': len(missing_lower), 'reverse': len(reverse), 'fdr': fdr, 'tpr': tpr,
              'fpr': fpr, 'shd': shd, 'nnz': pred_size, 'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore}
        for i in mt:
            mt[i] = round(mt[i], decimal_num)

        return mt

    # 用于计算估计图和真实图之间的 G 分数（G-Score）。
    @staticmethod
    def _cal_gscore(W_p, W_true):
        # W_p: 估计的 DAG 邻接矩阵，可以是 {0, 1, -1}。
        # W_true: 真实的 DAG 邻接矩阵，通常是 {0, 1}。
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """
        # 计算真实图中的有向边数量。
        num_true = W_true.sum(axis=1).sum()

        # true_positives
        # 计算估计图和真实图中的共同有向边数量。
        num_tp = (W_p + W_true).applymap(lambda elem: 1 if elem ==
                                         2 else 0).sum(axis=1).sum()
        # False Positives + Reversed Edges
        #  计算估计图中的错误有向边数量和反转边数量之和。
        num_fn_r = (W_p - W_true).applymap(lambda elem: 1 if elem ==
                                           1 else 0).sum(axis=1).sum()
        # 如果真实图中有向边数量不为零，计算 G 分数，即最大值函数，如果 num_tp - num_fn_r 为负，则取 0，否则取 (num_tp - num_fn_r) / num_true。
        if num_true != 0:
            score = np.max((num_tp-num_fn_r, 0))/num_true
        #     如果真实图中有向边数量为零，G 分数为 0。
        else:
            score = 0
        return score

    # _cal_precision_recall 方法用于计算估计图和真实图之间的精确度、召回率和 F1 分数
    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """
        #  断言估计图和真实图的形状相同，并且它们都是方阵。
        assert (W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
        # 计算估计图和真实图中的共同有向边数量（True Positives）。
        TP = (W_p + W_true).applymap(lambda elem: 1 if elem ==
                                     2 else 0).sum(axis=1).sum()
        # 计算估计图中的有向边数量（True Positives + False Positives）。
        TP_FP = W_p.sum(axis=1).sum()
        # 计算真实图中的有向边数量（True Positives + False Negatives）。
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/TP_FP
        recall = TP/TP_FN
        F1 = 2*(recall*precision)/(recall+precision)

        return precision, recall, F1

    # 用于计算估计图和真实图之间的一些特定指标。
    # forb_edges: 一个列表，包含禁止的边。
    # order: 一个列表，包含顺序约束的边。
    # ancs: 一个列表，包含祖先约束的边。
    # forb_ancs: 一个列表，包含禁止的祖先约束的边。
    # prior_exist: 一个列表，包含先验存在的边。
    def _count_forb_edge(self, B_est, forb_edges=None, order=None, ancs=None, forb_ancs=None, prior_exist=[]):
        # 检查 prior_exist['fe'][0] 是否为 True，这可能表示某些禁止边在估计图中是存在的。
        if prior_exist['fe'][0] == True:
            self.metrics["fe_st"] = 0
            self.metrics["fe_nost"] = 0
            for prior in forb_edges:
                if B_est[prior[0], prior[1]] == 0:
                    self.metrics["fe_st"] += 1
                else:
                    self.metrics["fe_nost"] += 1

        if prior_exist['o'][0] == True:
            self.metrics["o_st"] = 0
            self.metrics["o_nost"] = 0
            for prior in order:
                if check_path(B_est, prior[0], prior[1]):
                    self.metrics["o_st"] += 1
                else:
                    self.metrics["o_nost"] += 1

        if prior_exist['a'][0] == True:
            self.metrics["a_st"] = 0
            self.metrics["a_nost"] = 0
            for prior in ancs:
                if check_path(B_est, prior[0], prior[1]):
                    self.metrics["a_st"] += 1
                else:
                    self.metrics["a_nost"] += 1
        if prior_exist['fa'][0] == True:
            raise KeyError

    def _count_anc(self, B_est, ancs):
        self.metrics["ancs_statisfy"] = 0
        for prior in ancs:
            if check_path(B_est, prior[0], prior[1]):
                self.metrics["ancs_statisfy"] += 1

# 用于检查一个有向图（DAG）中是否存在从源节点到目标节点的有向路径。
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
