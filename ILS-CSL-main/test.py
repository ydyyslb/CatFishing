import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import chi2, norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from itertools import combinations, chain


# 生成一个可迭代对象（如列表或元组）的所有子集，包括空集和它本身。
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# 计算离散变量的 BIC
@njit(fastmath=True)
def bic(data, arities, cols):
    strides = np.empty(len(cols), dtype=np.int64)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[cols[idx]]
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ij = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ij = 0
        for i in range(len(cols)):
            idx_ijk += int(data[rowidx, cols[i]]) * strides[i]
            if i != 0:
                idx_ij += int(data[rowidx, cols[i]]) * strides[i]
        N_ijk[idx_ijk] += 1
        for i in range(arities[cols[0]]):
            N_ij[idx_ij + i * strides[0]] += 1
    bic_value = 0.0
    for i in range(stride):
        if N_ijk[i] != 0:
            bic_value += N_ijk[i] * np.log(N_ijk[i] / N_ij[i])
    bic_value -= 0.5 * np.log(data.shape[0]) * (arities[cols[0]] - 1) * strides[0]
    return bic_value


# 计算连续变量的 BIC
def bic_g(data, arities, cols):
    y = data[:, cols[0]]
    if len(cols) == 1:
        resids = y - np.mean(y)
    else:
        X = data[:, cols[1:]]
        reg = LinearRegression().fit(X, y)
        preds = reg.predict(X)
        resids = y - preds
    sd = np.std(resids)
    numparams = len(cols) + 1  # 包括截距和标准差
    bic_value = norm.logpdf(resids, scale=sd).sum() - np.log(data.shape[0]) / 2 * numparams
    return bic_value


# 计算离散目标与连续父变量的 BIC（基于逻辑回归）
def bic_logistic(data, arities, cols):
    y = data[:, cols[0]].astype(int)
    if len(cols) == 1:
        # 没有父节点时，预测基准概率
        p = np.mean(y)
        # 避免概率为0或1导致的log(0)问题，添加一个极小值
        p = np.clip(p, 1e-10, 1 - 1e-10)
        log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        numparams = 1  # 仅截距
    else:
        X = data[:, cols[1:]]
        reg = LogisticRegression(solver='liblinear').fit(X, y)
        preds = reg.predict_proba(X)
        # 避免概率为0或1导致的log(0)问题，添加一个极小值
        preds = np.clip(preds, 1e-10, 1 - 1e-10)
        log_likelihood = np.sum(y * np.log(preds[:, 1]) + (1 - y) * np.log(preds[:, 0]))
        numparams = len(cols[1:]) + 1  # 包括截距
    bic_value = -2 * log_likelihood + np.log(data.shape[0]) * numparams
    return bic_value


# 读取数据
def load_data(file_path):
    try:
        data_df = pd.read_csv(file_path)
        print("成功加载数据。")
        return data_df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


# 设置基数（arities）
def set_arities(data_df):
    arities = []
    for column in data_df.columns:
        unique_values = data_df[column].nunique()
        # 假设数据为连续变量，如果某列的唯一值小于或等于10，则认为是离散变量
        if unique_values <= 10:
            arities.append(unique_values)
        else:
            # 对于连续变量，设置基数为1
            arities.append(1)
    return np.array(arities, dtype=np.int64)


# 计算所有特征组合的 BIC 分数并保存
def compute_bic_scores(data_array, arities, save_path):
    num_features = data_array.shape[1]
    feature_indices = list(range(num_features))

    with open(save_path, 'w') as f:
        # 写入变量数量
        f.write(f"{num_features}\n")

        # 遍历每个变量作为目标变量
        for target_var in feature_indices:
            print(f"正在处理目标变量: {target_var}")
            # 根据目标变量类型选择可能的父节点
            if arities[target_var] == 1:
                # 连续目标变量，只选择连续父节点
                possible_parents = [i for i in feature_indices if i != target_var and arities[i] == 1]
            else:
                # 离散目标变量，父节点可以是连续或离散
                possible_parents = [i for i in feature_indices if i != target_var]

            num_models = 2 ** len(possible_parents)
            f.write(f"{target_var} {num_models}\n")

            # 遍历所有父节点组合
            for r in range(len(possible_parents) + 1):
                for parents in combinations(possible_parents, r):
                    cols = np.array([target_var] + list(parents), dtype=np.int64)

                    # 判断目标变量和父节点的类型
                    target_arity = arities[target_var]
                    parents_arities = arities[list(parents)]

                    if target_arity == 1 and len(parents_arities) > 0 and np.all(parents_arities == 1):
                        # 连续目标变量和连续父节点，使用 bic_g
                        score = bic_g(data_array, arities, cols)
                    elif target_arity > 1 and len(parents_arities) > 0 and np.all(parents_arities == 1):
                        # 离散目标变量和连续父节点，使用 bic_logistic
                        score = bic_logistic(data_array, arities, cols)
                    elif target_arity > 1 and len(parents_arities) > 0 and np.any(parents_arities > 1):
                        # 离散目标变量和离散父节点，使用 bic
                        score = bic(data_array, arities, cols)
                    elif target_arity == 1 and len(parents_arities) == 0:
                        # 连续目标变量且没有父节点，使用 bic_g
                        score = bic_g(data_array, arities, cols)
                    elif target_arity > 1 and len(parents_arities) == 0:
                        # 离散目标变量且没有父节点，使用 bic_logistic
                        score = bic_logistic(data_array, arities, cols)
                    else:
                        # 其他情况，默认为离散变量使用 bic
                        score = bic(data_array, arities, cols)

                    # 写入 BIC 分数及父节点信息
                    if len(parents) == 0:
                        f.write(f"{score} 0\n")
                    else:
                        parents_str = ' '.join(map(str, parents))
                        f.write(f"{score} {len(parents)} {parents_str}\n")
            print(f"已为变量 {target_var} 写入 BIC 分数")
    print(f"BIC分数已保存到 {save_path}")


def main():
    # 文件路径
    data_file= f"data/csv/UCI_500_1.csv"  # 替换为您的数据集路径
    save_file = 'bic_scores.txt'

    # 加载数据
    data_df = load_data(data_file)
    if data_df is None:
        return

    # 设置基数
    arities = set_arities(data_df)
    print(f"基数 (arities): {arities}")

    # 将数据转换为 NumPy 数组
    data_array = data_df.to_numpy()

    # 计算 BIC 分数并保存
    compute_bic_scores(data_array, arities, save_file)


if __name__ == "__main__":
    main()
