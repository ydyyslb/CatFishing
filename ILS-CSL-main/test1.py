import pandas as pd
import numpy as np
from itertools import combinations
from numba import njit
import os


@njit(fastmath=True)
def bic(data, arities, cols):
    strides = np.empty(len(cols), dtype=np.uint32)
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
            idx_ijk += data[rowidx, cols[i]] * strides[i]
            if i != 0:
                idx_ij += data[rowidx, cols[i]] * strides[i]
        N_ijk[idx_ijk] += 1
        for i in range(arities[cols[0]]):
            N_ij[idx_ij + i * strides[0]] += 1
    bic_score = 0.0
    for i in range(stride):
        if N_ijk[i] != 0:
            bic_score += N_ijk[i] * np.log(N_ijk[i] / N_ij[i])
    bic_score -= 0.5 * np.log(data.shape[0]) * (arities[cols[0]] - 1) * strides[0]
    return bic_score


def load_and_encode_data(kwargs):
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    data = pd.read_csv(filename, dtype='category')

    # 将类别变量编码为整数
    for col in data.columns:
        data[col] = data[col].cat.codes

    return data


def compute_bic_scores(data, target_variable, max_subset_size=3):
    num_columns = data.shape[1]
    columns = list(range(num_columns))
    columns.remove(target_variable)  # 排除目标变量
    arities = data.nunique().values  # 每个特征的类别数

    bic_results = []

    # 父变量的组合大小从0到max_subset_size
    for r in range(0, max_subset_size + 1):
        for subset in combinations(columns, r):
            subset_with_target = (target_variable,) + subset  # 将目标变量放在第一个
            bic_score = bic(data.values, arities, np.array(subset_with_target, dtype=np.int32))
            bic_results.append((subset_with_target, bic_score))
            print(f"目标变量 {target_variable}，特征组合 {subset_with_target} 的 BIC 分数: {bic_score}")

    return bic_results


def save_bic_scores(all_bic_scores, output_file):
    print(f"开始保存所有变量的 BIC 分数到 {output_file}")  # 调试信息
    with open(output_file, 'w') as f:
        num_variables = len(all_bic_scores)
        f.write(f"{num_variables}\n")
        for target_variable, bic_scores in all_bic_scores.items():
            f.write(f"{target_variable} {len(bic_scores)}\n")
            for subset, score in bic_scores:
                num_parents = len(subset) - 1  # 除去目标变量自身
                parents = [str(col) for col in subset if col != target_variable]
                f.write(f"{score} {num_parents} {' '.join(parents)}\n")
    print(f"所有变量的 BIC 分数已保存到 {output_file}")


def main():
    # 配置字典
    kwargs = {
        'd': 'Depression',  # 示例数据集名
        's': '500',       # 示例样本
        'r': '1'          # 示例运行标识
    }

    # 打印当前工作目录
    print(f"当前工作目录: {os.getcwd()}")

    # 加载并编码数据
    data = load_and_encode_data(kwargs)

    num_columns = data.shape[1]
    print(f"数据集包含 {num_columns} 个变量。")

    # 将所有变量作为目标变量
    target_variables = list(range(num_columns))

    # 定义输出目录
    output_dir = f"data/score/bic/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}"
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有变量的 BIC 分数
    all_bic_scores = {}

    # 遍历每个目标变量
    for target_variable in target_variables:
        print(f"正在处理目标变量 {target_variable}...")
        # 计算 BIC 分数，限制组合大小
        bic_scores = compute_bic_scores(data, target_variable, max_subset_size=3)  # 设置为3
        all_bic_scores[target_variable] = bic_scores

    # 定义输出文件路径
    output_file = f"{output_dir}/bic_score.txt"

    # 保存所有 BIC 分数
    save_bic_scores(all_bic_scores, output_file)

    print("所有变量的 BIC 分数计算完成。")


if __name__ == "__main__":
    main()
