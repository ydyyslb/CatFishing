from notears.notears.locally_connected import LocallyConnected
from notears.notears.lbfgsb_scipy import LBFGSBScipy
from notears.notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
from notears.notears import utils
import torch
from hc.accessory import local_score, local_score_from_storage
from hc.DAG import DAG, check_cycle
from utils import parse_parents_score
from utils import parse_parents_score, soft_constraint, write_parents_score
class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class NotearsSobolev(nn.Module):
    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - self.d  # 使用 self.d 替代 d
        # 另一种形式，稍快但代价是数值稳定性
        # M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, self.d - 1)
        # h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, score, prior_bonus, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    score_torch = torch.from_numpy(score)
    prior_bonus_torch = torch.from_numpy(prior_bonus)

    score_weight = 0.1
    prior_bonus_weight = 0.1

    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            score_term = torch.sum(score_torch * model.fc1_to_adj())  # 评分项
            prior_bonus_term = torch.sum(prior_bonus_torch * model.fc1_to_adj())  # 先验奖励项
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            # 组合所有项到目标函数中
            primal_obj = loss + penalty + l2_reg + l1_reg + score_weight * score_term + prior_bonus_weight * prior_bonus_term
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # 更新模型
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new



def dual_ascent_step_dict(model, X, score_dict, lambda1, lambda2, rho, alpha, h, rho_max, D, cache, candidate, dag):
    """使用预计算的分数字典执行增强拉格朗日法中的双重上升步骤。"""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    max_inner_iter = 100  # 设置内部最大迭代次数，避免无限循环

    for inner_iter in range(max_inner_iter):
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()

            # 计算先验得分项
            new_score_term = sum(
                cache[var][tuple(sorted(dag[var]['par']))]
                for var in D.varnames
            )

            primal_obj = loss + penalty + l2_reg + l1_reg - new_score_term
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()

        # 判断收敛条件
        if h_new <= 0.25 * h or rho >= rho_max:
            break
        else:
            rho *= 10
            alpha += rho * h_new

    alpha += rho * h_new  # 更新拉格朗日乘子
    return rho, alpha, h_new

# def dual_ascent_step_dict(model, X, score_dict, lambda1, lambda2, rho, alpha, h, rho_max):
#     """使用字典数据结构执行增强拉格朗日法中的双重上升步骤。"""
#     h_new = None
#     optimizer = LBFGSBScipy(model.parameters())
#     X_torch = torch.from_numpy(X)
#
#     score_weight = 0.5  # 根据需要调整
#
#     while rho < rho_max:
#         def closure():
#             optimizer.zero_grad()
#             X_hat = model(X_torch)
#             loss = squared_loss(X_hat, X_torch)
#             h_val = model.h_func()
#             new_score_term = 0
#
#             # 使用字典结构计算新的得分项
#             for var in score_dict:
#                 for new_score, _ in score_dict[var]:
#                     new_score_term += new_score
#
#             penalty = 0.5 * rho * h_val * h_val + alpha * h_val
#             l2_reg = 0.5 * lambda2 * model.l2_reg()
#             l1_reg = lambda1 * model.fc1_l1_reg()
#
#             # 组合所有项到目标函数中
#             primal_obj = loss + penalty + l2_reg + l1_reg - score_weight * new_score_term
#             primal_obj.backward()
#             return primal_obj
#
#         optimizer.step(closure)  # 更新模型
#         with torch.no_grad():
#             h_new = model.h_func().item()
#         if h_new > 0.25 * h:
#             rho *= 10
#         else:
#             break
#     alpha += rho * h_new
#     return rho, alpha, h_new




def notears_nonlinear_dict(model: nn.Module,
                           X: np.ndarray,
                           D: DAG,
                           score_filepath: str,
                           lambda1: float = 0.,
                           lambda2: float = 0.,
                           max_iter: int = 50,
                           h_tol: float = 1e-8,
                           rho_max: float = 1e+16,
                           w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf

    # 初始化分数字典
    score_dict = parse_parents_score(score_filepath) if score_filepath else {}
    print(f"分数字典初始化完成，共有 {len(score_dict)} 条目。")

    # 初始化候选集和DAG结构
    candidate = {}
    dag = {}
    cache = {}
    record_score = 0

    for var in D.varnames:
        # 初始化候选父节点
        candidate[var] = [v for v in D.varnames if v != var] if D.pc is None else list(D.pc[var])
        dag[var] = {'par': [], 'nei': []}
        cache[var] = {}

        # 初始化空父节点的分数
        cols = [D.varnames.index(var)]
        cache[var][tuple()] = local_score_from_storage(cols, score_dict) if score_filepath else local_score(D.data, D.arities, cols, D.score)
        record_score += 1

    # 处理先验信息
    for var in D.varnames:
        # 添加先验父节点
        prior_parents = D.a_prior[var]['par']
        dag[var]['par'].extend(prior_parents)
        dag[var]['par'] = sorted(set(dag[var]['par']))

        # 更新缓存
        par_sea = tuple(dag[var]['par'])
        if par_sea not in cache[var]:
            cols = [D.varnames.index(v) for v in (var,) + par_sea]
            cache[var][par_sea] = local_score_from_storage(cols, score_dict)
            record_score += 1

        # 更新候选集，移除先验父节点
        candidate[var] = [c for c in candidate[var] if c not in prior_parents]

        # 处理反向先验信息
        for par in D.r_prior[var]['par']:
            dag[par]['par'].append(var)
            dag[par]['par'] = sorted(set(dag[par]['par']))

            par_sea_par = tuple(dag[par]['par'])
            if par_sea_par not in cache[par]:
                cols_par = [D.varnames.index(v) for v in (par,) + par_sea_par]
                cache[par][par_sea_par] = local_score_from_storage(cols_par, score_dict)
                record_score += 1

            # 更新候选集，移除反向先验节点
            candidate[var] = [c for c in candidate[var] if c != par]
            candidate[par] = [c for c in candidate[par] if c != var]

        # 处理直接先验信息
        for par in D.d_prior[var]['par']:
            candidate[var] = [c for c in candidate[var] if c != par]
            candidate[par] = [c for c in candidate[par] if c != var]

    print(f"DAG结构和候选集初始化完成，共记录了 {record_score} 个分数。")

    # 主要迭代过程
    for iteration in range(max_iter):
        rho, alpha, h = dual_ascent_step_dict(model, X, score_dict, lambda1, lambda2, rho, alpha, h, rho_max, D, cache, candidate, dag)
        print(f"迭代 {iteration + 1}: h = {h}, rho = {rho}")
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est





# def notears_nonlinear_dict(model: nn.Module,
#                            X: np.ndarray,
#                            D: DAG,
#                            score_filepath=None,
#                            lambda1: float = 0.,
#                            lambda2: float = 0.,
#                            max_iter: int = 100,
#                            h_tol: float = 1e-8,
#                            rho_max: float = 1e+16,
#                            w_threshold: float = 0.3):
#     rho, alpha, h = 1.0, 0.0, np.inf
#     '''
#         :p aram data：用于学习BN（numpy数组）的训练数据
#         :p aram arities：每个变量的不同值的数量
#         :p aram varnames：变量名称
#         :p aram pc：为每个变量设置的候选父母和孩子
#         :p aram score：评分函数，包括：
#                        bic（离散变量的贝叶斯信息准则）
#                        bic_g（连续变量的贝叶斯信息准则）
#
#         ：return：学习的BN（bnlearn格式）
#         '''
#     # 初始化一个空字典 score_dict，用于存储预计算的分数。
#     score_dict = {}
#     # 如果提供了 score_filepath，则从该文件中解析预计算的分数。
#     if score_filepath is not None:
#         score_dict = parse_parents_score(score_filepath)
#
#     # 初始化一个变量 record_score，用于记录函数执行过程中调用评分函数的次数。
#     record_score = 0
#
#     # 定义一个内部函数 update_cache，用于更新存储在 cache 中的分数。这个函数在后续的代码中没有直接调用。
#     def update_cache(cache, var, par_sea):
#         if par_sea not in cache[var]:
#             cols = [D.varnames.index(x) for x in (var,) + par_sea]
#             cache[var][par_sea] = local_score_from_storage(
#                 cols, score_dict)
#
#     #  设置评分函数：
#     # 如果 D.score 是默认值，根据 D.arities 是否为 None 来决定使用 bic 还是 bic_g。
#     if D.score == 'default':
#         D.score = 'bic_g' if D.arities is None else 'bic'
#     # initialize the candidate parents-set for each variable
#     # 初始化候选父节点集合和DAG结构：
#     # 遍历每个变量，初始化其候选父节点集合和DAG中的父节点列表。
#     # 如果提供了 score_filepath，则从预计算分数中读取初始分数，否则调用评分函数计算。
#     candidate = {}
#     dag = {}
#     cache = {}
#     for var in D.varnames:
#         # Edge forbidden
#         # 如果 D.pc（条件概率表）为 None，说明没有父节点信息，因此每个变量的候选边列表是所有其他变量的名称，除了自己。
#         if D.pc is None:
#             candidate[var] = list(D.varnames)
#             candidate[var].remove(var)
#         # 如果 D.pc 不为 None，说明有父节点信息，因此每个变量的候选边列表是 D.pc 中与当前变量相关的父节点列表。
#         else:
#             candidate[var] = list(D.pc[var])
#         # 创建一个空字典 dag[var]，用于存储与变量 var 相关的父节点列表和邻居节点列表。
#         # 创建一个空列表 dag[var]['par']，用于存储变量 var 的父节点列表。
#         # 创建一个空列表 dag[var]['nei']，用于存储变量 var 的邻居节点列表。
#         # 创建一个空字典 cache[var]，用于存储与变量 var 相关的局部分数缓存。
#         dag[var] = {}
#         dag[var]['par'] = []
#         dag[var]['nei'] = []
#         cache[var] = {}
#         record_score += 1
#         # 如果提供了分数文件路径 score_filepath，说明分数可以从文件中读取。
#         if score_filepath is not None:
#             cache[var][tuple([])] = local_score_from_storage(
#                 [D.varnames.index(var)], score_dict)
#         # 如果 score_filepath 为 None，说明分数需要计算。
#         else:
#             cache[var][tuple([])] = local_score(
#                 D.data, D.arities, [D.varnames.index(var)], D.score)
#     for var in D.varnames:
#         # 添加先验信息：
#         # 将先验信息中的父节点添加到变量 var 的父节点列表 dag[var]['par'] 中，并对其进行排序。
#         dag[var]['par'] = sorted(dag[var]['par'] + D.a_prior[var]['par'])
#         par_sea = tuple(sorted(dag[var]['par']))
#         # 如果元组 par_sea 不在变量 var 的缓存中，说明这是第一次添加这个父节点集合，需要计算局部分数。
#         if par_sea not in cache[var]:
#             # 创建一个包含所有相关变量索引的列表 cols，这些变量包括 var 和它的父节点集合。
#             cols = [D.varnames.index(x) for x in (var,) + par_sea]
#             record_score += 1
#             # 从文件中读取局部分数，并将其存储在 cache[var] 的 par_sea 键下。
#             cache[var][par_sea] = local_score_from_storage(cols, score_dict)
#         # 循环遍历先验信息中的父节点 par。
#         for par in D.a_prior[var]['par']:
#             # 如果变量 var 在父节点 par 的候选父节点集合中，将其移除。
#             if par in candidate[var]:
#                 candidate[var].remove(par)
#             # 如果变量 var 在父节点 par 的候选父节点集合中，将其移除。
#             if var in candidate[par]:
#                 candidate[par].remove(var)
#         # 循环遍历先验信息中的反向先验信息中的父节点 par。
#         for par in D.r_prior[var]['par']:
#             # 将变量 var 添加到父节点 par 的父节点列表 dag[par]['par'] 中，并对其进行排序。
#             dag[par]['par'] = sorted(dag[par]['par'] + [var])
#             par_sea = tuple(sorted(dag[par]['par']))
#             # 如果元组 par_sea 不在父节点 par 的缓存中，说明这是第一次添加这个父节点集合，需要计算局部分数。
#             if par_sea not in cache[par]:
#                 # 创建一个包含所有相关变量索引的列表 cols。
#                 cols = [D.varnames.index(x) for x in (par,) + par_sea]
#                 record_score += 1
#                 cache[par][par_sea] = local_score_from_storage(cols, score_dict)
#             # 从变量 var 的候选父节点集合中移除父节点 par。
#             candidate[var].remove(par)
#             # 从父节点 par 的候选父节点集合中移除变量 var。
#             candidate[par].remove(var)
#         # 循环遍历先验信息中的直接先验信息中的父节点 par。
#         for par in D.d_prior[var]['par']:
#             candidate[var].remove(par)
#             candidate[par].remove(var)
#     for _ in range(max_iter):
#         # 使用新的双重上升步骤，使用字典数据结构
#         rho, alpha, h = dual_ascent_step_dict(model, X, score_dict, prior_bonus, lambda1, lambda2, rho, alpha, h, rho_max)
#         if h <= h_tol or rho >= rho_max:
#             break
#     W_est = model.fc1_to_adj()
#     W_est[np.abs(W_est) < w_threshold] = 0
#     return W_est
def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      score: np.ndarray,  # 新增参数: 评分矩阵
                      prior_bonus: np.ndarray,  # 新增参数: 先验奖励矩阵
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, score, prior_bonus, lambda1, lambda2, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est



def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import notears.notears.utils as ut
    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('W_true.csv', B_true, delimiter=',')

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)


if __name__ == '__main__':
    main()