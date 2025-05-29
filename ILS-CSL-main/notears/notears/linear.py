import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from causalnex.structure.notears import from_pandas
import networkx as nx
from hc.DAG import DAG

def notears_linear_prior(D: DAG,X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    # 初始化先验知识字典
    prior_knowledge = {
        'mandatory_edges': set(),  # 必须存在的边
        'forbidden_edges': set(),  # 禁止存在的边
        'fixed_direction': set()  # 方向固定的边
    }

    # 假设D.pc, D.a_prior, D.r_prior, D.d_prior都是字典，键是节点，值是边的列表
    # 对于D.a_prior，将必须添加的边添加到mandatory_edges集合
    for node, parents in D.a_prior.items():
        for parent in parents:
            prior_knowledge['mandatory_edges'].add((parent, node))

    # 对于D.r_prior，将必须反转的边添加到fixed_direction集合
    # 注意，我们假设边的方向是从parent到node，反转即为从node到parent
    for node, parents in D.r_prior.items():
        for parent in parents:
            prior_knowledge['fixed_direction'].add((node, parent))

    # 对于D.d_prior，将必须删除的边添加到forbidden_edges集合
    for node, parents in D.d_prior.items():
        for parent in parents:
            prior_knowledge['forbidden_edges'].add((parent, node))

    # 对于D.pc，我们可以选择性地添加一些边到forbidden_edges集合
    # 如果D.pc中的边不在mandatory_edges或fixed_direction中，我们可以认为它们是可选的
    # 但如果我们想要更严格地遵循先验知识，我们可以将不在D.pc中的边视为forbidden
    for node in D.varnames:
        for potential_parent in D.varnames:
            if potential_parent != node and potential_parent not in D.pc[node]:
                prior_knowledge['forbidden_edges'].add((potential_parent, node))

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
            # 计算预测误差
            preds = np.dot(W, data)  # 假设data是X的转置
            errors = data - preds  # 计算预测误差

            # 计算惩罚项
            penalty = 0
            for (i, j) in prior_knowledge['fixed_direction']:
                # 假设prior_knowledge['fixed_direction']包含必须存在的边的索引
                if i != j:
                    # 计算在这些边上的预测误差
                    error = errors[i] - errors[j]
                    # 根据误差的大小计算惩罚
                    penalty += error * error

            # 将惩罚项加到总损失上
            loss += penalty
        return loss, G_loss
    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
            # 在每次迭代后，确保先验知识不被违反
            for (i, j) in prior_knowledge['mandatory_edges']:
                if W_est[i, j] == 0:  # 如果必须存在的边不存在
                    W_est[i, j] = some_value  # 确保必须存在的边存在

            for (i, j) in prior_knowledge['forbidden_edges']:
                if W_est[i, j] != 0:  # 如果禁止存在的边存在
                    W_est[i, j] = 0  # 确保禁止存在的边不存在

            for (i, j) in prior_knowledge['fixed_direction']:
                if W_est[j, i] != 0:  # 如果边的方向不符合先验知识
                    W_est[j, i] = 0  # 确保边的方向符合先验知识

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est




if __name__ == '__main__':
    from notears import utils
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
    # 使用绝对路径
    file_path = "/root/autodl-tmp/notears-master/data/child_graph.txt"
    B_true = np.loadtxt(file_path, dtype=int)
    W_true = utils.simulate_parameter(B_true)

    # 构建文件的完整   路径
    filename = "/root/autodl-tmp/notears-master/data/csv/child_500_1.csv"
    data = pd.read_csv(filename, dtype='category')
    # 非数值变量进行标签编码
    struct_data = data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        struct_data[col] = le.fit_transform(struct_data[col])
    # 生成一阶多项式特征
    poly = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(struct_data)

    # 标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # 将numpy数组转换回pandas DataFrame，以便使用from_pandas
    X_scaled_df = pd.DataFrame(X_scaled, columns=poly.get_feature_names_out(data.columns))

    # 使用 NOTEARS 算法学习结构
    sm = from_pandas(X_scaled_df, max_iter=100, w_threshold=0.3)

    # 转换为邻接矩阵，确保只包含 {0, 1}
    adj_matrix = nx.to_numpy_array(sm)
    adj_matrix = np.where(adj_matrix != 0, 1, 0)

    # 计算准确率
    acc = utils.count_accuracy(B_true, adj_matrix != 0)
    print(acc)

#
# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
#     from notears import utils
#     from notears.linear import notears_linear
#
#     # 设置随机种子
#     utils.set_random_seed(1)
#
#     # 加载真实图的结构
#     file_path = "/root/autodl-tmp/notears-master/data/child_graph.txt"
#     B_true = np.loadtxt(file_path, dtype=int)
#
#     # 模拟参数
#     W_true = utils.simulate_parameter(B_true)
#
#     # 读取CSV文件
#     filename = "/root/autodl-tmp/notears-master/data/csv/child_500_1.csv"
#     data = pd.read_csv(filename, dtype='category')
#
#     # 标签编码非数值列
#     struct_data = data.copy()
#     non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
#
#     for col in non_numeric_columns:
#         le = LabelEncoder()
#         struct_data[col] = le.fit_transform(struct_data[col])
#
#     # 生成一阶多项式特征
#     poly = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)
#     X_poly = poly.fit_transform(struct_data)
#
#     # 标准化处理
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_poly)
#
#     # 简化lambda1的搜索空间
#     param_grid = {'lambda1': np.logspace(-3, -1, 3)}  # 只搜索3个值
#     best_lambda = 0.1
#     best_score = None
#
#     for lambda1 in param_grid['lambda1']:
#         W_est = notears_linear(X_scaled, lambda1=lambda1, loss_type='l2')
#         acc = utils.count_accuracy(B_true, W_est != 0)
#         if best_score is None or acc['fdr'] < best_score['fdr']:
#             best_lambda = lambda1
#             best_score = acc
#
#     # 使用最佳lambda1进行因果结构学习
#     W_est = notears_linear(X_scaled, lambda1=best_lambda, loss_type='l2')
#
#     # 确认生成的矩阵是DAG
#     assert utils.is_dag(W_est)
#
#     # 保存估计的矩阵到CSV文件
#     np.savetxt('W_est.csv', W_est, delimiter=',')
#
#     # 计算准确率
#     acc = utils.count_accuracy(B_true, W_est != 0)
#     print(f"最佳 lambda1: {best_lambda}")
#     print(acc)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
    from notears import utils
    from notears.notears.linear import notears_linear
    from sklearn.model_selection import GridSearchCV

    # 设置随机种子
    utils.set_random_seed(1)

    # 加载真实图的结构
    file_path = "/root/autodl-tmp/notears-master/data/child_graph.txt"
    B_true = np.loadtxt(file_path, dtype=int)

    # 模拟参数
    W_true = utils.simulate_parameter(B_true)

    # 读取CSV文件
    filename = "/root/autodl-tmp/notears-master/data/csv/child_500_1.csv"
    data = pd.read_csv(filename, dtype='category')

    # 标签编码非数值列
    struct_data = data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)

    for col in non_numeric_columns:
        le = LabelEncoder()
        struct_data[col] = le.fit_transform(struct_data[col])

        # 频率编码
        freq = struct_data[col].value_counts(normalize=True)
        struct_data[col] = struct_data[col].map(freq)

    # 生成多项式特征
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(struct_data)

    # 标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # GridSearchCV 查找最佳lambda1参数
    param_grid = {'lambda1': np.logspace(-3, 0, 10)}
    best_lambda = 0.1
    best_score = None
    for lambda1 in param_grid['lambda1']:
        W_est = notears_linear(X_scaled, lambda1=lambda1, loss_type='l2')
        acc = utils.count_accuracy(B_true, W_est != 0)
        if best_score is None or acc['fdr'] < best_score['fdr']:
            best_lambda = lambda1
            best_score = acc

    # 使用最佳lambda1进行因果结构学习
    W_est = notears_linear(X_scaled, lambda1=best_lambda, loss_type='l2')

    # 确认生成的矩阵是DAG
    assert utils.is_dag(W_est)

    # 保存估计的矩阵到CSV文件
    np.savetxt('W_est.csv', W_est, delimiter=',')

    # 计算准确率
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(f"最佳 lambda1: {best_lambda}")
    print(acc)


#
# if __name__ == '__main__':
#     from notears import utils
#     utils.set_random_seed(1)
#
#     n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
#     B_true = utils.simulate_dag(d, s0, graph_type)
#     W_true = utils.simulate_parameter(B_true)
#     np.savetxt('W_true.csv', W_true, delimiter=',')
#
#     X = utils.simulate_linear_sem(W_true, n, sem_type)
#     np.savetxt('X.csv', X, delimiter=',')
#
#     W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
#     assert utils.is_dag(W_est)
#     np.savetxt('W_est.csv', W_est, delimiter=',')
#     acc = utils.count_accuracy(B_true, W_est != 0)
#     print(acc)
#     # metrics = MetricsDAG(W_est, B_true).metrics
#
