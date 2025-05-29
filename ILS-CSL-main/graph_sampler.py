
import numpy as np

from main import Iteration_CSL
from utils import *

# 这个类似乎用于处理与图相关的采样操作，特别是在因果结构学习（Causal Structure Learning）的上下文中。
class GraphSampler:
    # ev_dag 参数是一个矩阵，表示估计的有向无环图（DAG）。
    # i2p 参数是一个字典，将节点编号映射到节点名称。如果未提供，将使用 self.i2p。
    # dataset 参数是一个字符串，表示数据集的名称。
    def __init__(self, ev_dag, i2p=None, dataset="dataset"):
        # 初始化实例变量
        # 实例变量 self.i2p 被初始化为提供的 i2p 字典，如果未提供，则使用 self.i2p。
        # 实例变量 self.ev_dag 被初始化为提供的 ev_dag 矩阵。
        # 实例变量 self.study_dir 被初始化为包含数据集名称的目录路径，用于存储采样结果。
        # 使用 os.makedirs 函数创建 self.study_dir 目录，如果目录已存在，则不抛出异常。
        self.i2p = i2p if i2p is not None else self.i2p
        self.ev_dag = ev_dag
        self.study_dir = f"pairwise_samples/{dataset}"
        os.makedirs(self.study_dir, exist_ok=True)

    # 计算估计的有向无环图中所有节点对之间的可达性。
    def floyd_warshall_reachability(self):
        # 初始化可达性矩阵
        # 初始化节点数量 n 为 self.ev_dag 矩阵的行数或列数。
        # 创建一个可达性矩阵 reachability，其大小与 self.ev_dag 相同，并将其初始化为 self.ev_dag 的副本。
        n = len(self.ev_dag)
        reachability = self.ev_dag.copy()
        # Floyd-Warshall算法
        # 使用Floyd-Warshall算法来更新可达性矩阵。
        # 算法的基本思想是，如果节点 i 和 j 之间直接可达，或者节点 i 经过节点 k 可达节点 j，则节点 i 和 j 之间是可达的。
        # 遍历所有可能的节点对 (i, j)，对于每个节点 k，检查 reachability[i, j] 是否可以通过 reachability[i, k] 和 reachability[k, j] 更新。
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    reachability[i, j] = reachability[i, j] or (
                        reachability[i, k] and reachability[k, j])
        #  返回可达性矩阵
        return reachability

    # 返回估计的有向无环图中直接连接的所有节点对。
    def _direct_connections(self):
        # 获取直接连接的节点对
        # 使用 np.where 函数来找到 self.ev_dag 矩阵中所有大于0的元素的位置，这些元素表示存在边。
        # rows 和 cols 变量分别存储了行的索引和列的索引。
        # 使用 zip 函数将行和列的索引配对，形成所有直接连接的节点对。
        # 最后，将节点对列表转换为元组列表，并返回
        rows, cols = np.where(self.ev_dag > 0)
        return list(zip(rows, cols))

    # 返回估计的有向无环图中间接连接的所有节点对。
    def _indirect_connections(self):
        # 计算可达性矩阵
        # 调用 self.floyd_warshall_reachability 方法计算可达性矩阵。这个矩阵将包含图中所有节点对之间的可达性信息。

        # 获取间接连接的节点对
        # 使用 np.where 函数来找到可达性矩阵 reachability_matrix 与原始估计图 self.ev_dag 之间的差异，这些差异表示通过其他节点间接连接的边。
        # rows 和 cols 变量分别存储了行的索引和列的索引。
        # 使用 zip 函数将行和列的索引配对，形成所有间接连接的节点对。
        # 最后，将节点对列表转换为元组列表，并返回。
        reachability_matrix = self.floyd_warshall_reachability()
        rows, cols = np.where((reachability_matrix - self.ev_dag) > 0)
        return list(zip(rows, cols))

    # 返回估计的有向无环图中没有直接或间接连接的所有节点对。
    def _no_connections(self):
        # 计算可达性矩阵
        # 这个矩阵将包含图中所有节点对之间的可达性信息。
        reachability_matrix = self.floyd_warshall_reachability()
        # 初始化没有连接的节点对列表，用于存储没有直接或间接连接的节点对。
        no_connection_pairs = []
        # 遍历所有节点对:
        # 对于每一对节点 (i, j)，检查它们之间是否有直接或间接连接。
        # 如果没有连接（即 reachability_matrix[i, j] 和 reachability_matrix[j, i] 都为0），则将它们添加到 no_connection_pairs 列表中。
        n = reachability_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if reachability_matrix[i, j] == 0 and reachability_matrix[j, i] == 0:
                    no_connection_pairs.append((i, j))
        # 回没有连接的节点对列表
        return no_connection_pairs

    # 从估计的有向无环图中采样不同类型的节点对。
    def sample_pairs(self, num_samples=20):
        # num_samples 参数是一个整数，表示要采样的节点对数量。默认为20。
        # 定义采样方法字典:其中键是采样类型的名称，值是相应采样方法的引用。
        sampling_methods = {
            "direct_connections": self._direct_connections,
            "indirect_connections": self._indirect_connections,
            "no_connection": self._no_connections
        }
        # 初始化采样结果字典,用于存储不同类型节点对的采样结果。
        sampled_results = {}
        # 循环采样不同类型的节点对
        # 对于字典中的每个键值对，调用相应的采样方法获取节点对列表 pairs。
        # 计算要采样的数量 num，它等于 num_samples 和 len(pairs) 中的较小值。
        # 使用 np.random.choice 函数从 pairs 中随机选择 num 个节点对，不进行替换。
        # 将选中的节点对存储在 sampled_results 字典中，并按升序排序。
        # 使用 rprint 函数（可能是自定义的）打印采样结果。
        for key, method in sampling_methods.items():
            pairs = method()
            num = min(num_samples, len(pairs))
            sampled_indices = np.random.choice(
                len(pairs), num, replace=False)
            sampled_pairs = [pairs[i] for i in sampled_indices]
            sampled_results[key] = sorted(sampled_pairs)
            rprint(f"{key.capitalize()} connections: {len(pairs)}, Sampled: {num}")

        # 保存采样结果到文件
        # 对于字典中的每个键值对，将采样结果保存到两个文件中：一个是包含原始节点编号的文件，另一个是包含节点名称的文件。
        # write_txt 函数（可能是自定义的）用于将内容写入文本文件。
        for key, pairs in sampled_results.items():
            write_txt(f"{self.study_dir}/{key}.txt",
                      "\n".join([f"{pair[0]},{pair[1]}" for pair in pairs]))
            write_txt(f"{self.study_dir}/{key}_str.txt",
                      "\n".join([f"{self.i2p[pair[0]]},{self.i2p[pair[1]]}" for pair in pairs]))
        # 返回采样结果字典
        return sampled_results


# 入口点
if __name__ == "__main__":
    # 循环遍历数据集
    for dataset in ["alarm", "asia", "insurance", "mildew", "child", "cancer", "water", "barley"]:
        # 创建 Iteration_CSL 实例
        icsl = Iteration_CSL(dataset)
        # 获取真实的有向无环图（DAG）
        ev_dag = icsl.true_dag  # adjacency matrix
        from rich import print as rprint
        # 创建 GraphSampler 实例,并传入真实的有向无环图的邻接矩阵和节点编号到名称的映射。
        sampler = GraphSampler(ev_dag,icsl.i2p,dataset)
        # 采样节点对:存储采样结果在 sampled_results 变量中。
        sampled_results = sampler.sample_pairs(20)
