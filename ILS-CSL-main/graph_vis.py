import pydot
from rich import print as rprint

from utils import *

# 表示和可视化图（Graph）
class GraphVis:
    # 构造方法，用于初始化 GraphVis 类的实例
    # ode_list: 一个列表，包含图中的节点（或顶点）。
    # edge_list: 一个列表，包含图中的边。每条边可能是一个元组或列表，包含两个节点的索引或节点本身。
    def __init__(self, node_list: list = None, edge_list: list = None):
        # correct2style: 一个字典，将布尔值（字符串形式）映射到线条样式。例如，“true” 对应 “solid”（实线），“false” 对应 “dashed”（虚线）。
        # prior2color: 一个字典，将边的优先级映射到颜色。例如，“exist” 对应 “red”（红色），“frob” 对应 “blue”（蓝色），“normal” 对应 “black”（黑色）。
        # nodes: 一个列表，存储图中的节点。
        # edges: 一个列表，存储图中的边。
        # i2p: 一个字典，将节点的索引映射到节点本身。例如，如果节点列表是 [A, B, C]，那么 i2p 将是 {0: A, 1: B, 2: C}。
        # p2i: 一个字典，将节点映射到节点的索引。例如，如果节点列表是 [A, B, C]，那么 p2i 将是 {A: 0, B: 1, C: 2}。
        self.correct2style = {"true": "solid", "false": "dashed"}
        self.prior2color = {"exist": "red", "frob": "blue", "normal": "black"}
        self.nodes = node_list
        self.edges = edge_list
        self.i2p = {i: p for i, p in enumerate(node_list)}
        self.p2i = {p: i for i, p in enumerate(node_list)}

    # 初始化图中的一些辅助边信息
    def init_aux_edges(self, exist_edges, forb_edges, true_edges):
        # exist_edges: 一个列表，包含图中存在的边。这些边可能表示图中实际连接的节点对，或者是在某些条件下必须存在的边。
        # forb_edges: 一个列表，包含图中禁止的边。这些边可能表示在图中不允许出现的连接，或者是在某些条件下不应该存在的边。
        # true_edges: 一个列表，包含图中为“真”的边。这些边可能表示在某种逻辑或条件验证下被确认为真实的连接。
        self.exist_edges = exist_edges
        self.forb_edges = forb_edges
        self.true_edges = true_edges

    # 生成图中每条边的概要信息
    def summary_info(self):
        # : 初始化一个空列表 info_list，用于存储每条边的概要信息。
        info_list = []
        #  遍历 self.edges 列表中的每一条边。self.edges 应该是 GraphVis 实例中的一个属性，它存储了图中所有的边。
        for edge in self.edges:
            # correct 和 prior。这两个值分别代表边的正确性和优先级。
            correct, prior = self._edge_type(edge)
            # : 创建一个字典 edge_info，它包含以下键值对：
            # "edge": 边的信息，通常是一个元组或列表，包含两个节点的索引或节点本身。
            # "correct": 表示边是否正确的布尔值或相应的描述。
            # "prior": 表示边优先级的字符串或相应的描述。
            edge_info = {"edge": edge, "correct": correct, "prior": prior}
            # 将 edge_info 字典添加到 info_list 列表中。
            info_list.append(edge_info)
        return info_list


    # 这个方法通过遍历节点和边的信息列表，使用 Pydot 库构建了一个有向图，并为图中的每条边设置了颜色和样式。这个有向图对象可以被用来生成 PNG、SVG 或其他格式的图片，以可视化地表示图的结构。
    # edge_info_list，这是一个包含每条边概要信息的列表
    def construct_digraph(self, edge_info_list):
        #  创建一个新的 pydot.Dot 对象，并指定图类型为 'digraph'，表示这是一个有向图。
        graph = pydot.Dot(graph_type='digraph')
        # 遍历 self.nodes 列表中的每个节点，
        for node in self.nodes:
            # 为图添加一个新的节点，使用 pydot.Node 创建节点对象。
            graph.add_node(pydot.Node(node))
        # 遍历 edge_info_list 列表中的每个元素，每个元素是一个包含边信息的字典，
        for edge_info in edge_info_list:
            # 从字典中提取边的定义，通常是一个包含两个节点的列表或元组。
            edge = edge_info["edge"]
            # 使用 self.prior2color 字典将边的优先级转换为颜色。
            color = self.prior2color[edge_info["prior"]]
            # 使用 self.correct2style 字典将边的正确性转换为线条样式。self.correct2style 也是在 GraphVis 类的构造方法中定义的，它将布尔值映射到线条样式。
            style = self.correct2style[edge_info["correct"]]
            # 为图添加一个新的边，使用 pydot.Edge 创建边对象，并设置边的起点、终点、颜色和样式。
            graph.add_edge(pydot.Edge(
                edge[0], edge[1], color=color, style=style))
        return graph

    # 它用于确定一条边的正确性和优先级
    def _edge_type(self, edge):
        #  检查当前边是否在 self.true_edges 列表中。self.true_edges 是一个存储图中为“真”的边的列表。那么这条边被标记为“正确“
        if edge in self.true_edges:  # edge in the true dag
            correct = "true"
        #     这条边被标记为“不正确”
        else:  # edge not in the true dag
            correct = "false"
        #  创建一个 reversed_edge，它是传入边 edge 的反向版本。例如，如果 edge 是 [A, B]，那么 reversed_edge 就是 [B, A]。
        reversed_edge = [edge[1], edge[0]]
        # 检查反向边是否在 self.exist_edges 列表中（表示必须存在的边），或者当前边是否在 self.forb_edges 列表中（表示禁止的边）。
        if reversed_edge in self.exist_edges or edge in self.forb_edges:  # B or C
            # 如果反向边在 self.exist_edges 和 self.true_edges 中，那么这条边被标记为“存在”，prior 被设置为 "exist"。
            if reversed_edge in self.exist_edges and reversed_edge in self.true_edges:
                prior = "exist"
            #     如果当前边在 self.forb_edges 中但不在 self.true_edges 中，那么这条边被标记为“禁止”，prior 被设置为 "frob"。
            elif edge in self.forb_edges and edge not in self.true_edges:
                prior = "frob"
            else:
                prior = "normal"
        else:
            prior = "normal"
        return correct, prior

    # 生成和保存图的视觉表示。这个方法完成了图的构建、可视化以及保存为图片文件的任务，并返回了更新后的边信息列表。
    # png_path 和 svg_path，它们分别表示生成的 PNG 图片和 SVG 图片的保存路径。
    def visualize(self, png_path: str = 'graph.png', svg_path: str = None):
        # 调用 summary_info 方法获取图中每条边的概要信息，并将结果存储在 edge_info_list 变量中。
        edge_info_list = self.summary_info()
        # 调用 construct_digraph 来构建一个有向图（digraph），这个方法可能使用 edge_info_list 中的信息来创建图，并将结果存储在 graph 变量中。
        graph = self.construct_digraph(edge_info_list)
        # 使用 Graphviz 库（或其他类似的库）将构建好的图保存为 PNG 图片，文件路径由 png_path 参数指定。
        graph.write_png(png_path)
        # 如果提供了 svg_path 参数，将图保存为 SVG 图片，文件路径由 svg_path 参数指定
        if svg_path:
            graph.write_svg(svg_path)
        #     输出一条消息，通知用户图已经被保存到指定的路径。这条消息包含了文件路径，并且使用了颜色和样式来增强可读性。
        rprint(
            f"📊 [bold yellow]Graphviz[/bold yellow] saved to [italic blue]{png_path}[/italic blue] and [italic blue]{svg_path}[/italic blue]")
        # edge_info string to nubmer
        # 将边信息中的节点名称转换为它们的索引。这是通过使用 self.p2i 字典（节点到索引的映射）来完成的。
        for edge_info in edge_info_list:
            edge_info["edge"] = [self.p2i[p] for p in edge_info["edge"]]
        # 返回更新后的 edge_info_list，其中边的节点名称已经被替换为相应的索引。
        return edge_info_list

# 它将一个邻接矩阵（表示有向无环图，DAG）转换为边的列表。
# ev_dag: 一个二维数组（例如 NumPy 数组），表示有向图的邻接矩阵。
# i2p:一 个 字典，将节点的索引映射到节点的名称。
# string: 一个布尔值，表示是否将边的索引转换为节点名称。默认为 True。
def matrix2edge(ev_dag, i2p, string=True):
    """
    output the edges of ev_dag in the format of [A, B],
    where A and B are the names of the nodes
    """
    #  初始化一个空列表 candidate_edges，用于存储图中的边。
    # : 检查邻接矩阵中位于 (i, j) 的元素是否为 1。 如果存在这样的边，将边的索引 [i, j] 添加到 candidate_edges 列表中。
    candidate_edges = []
    #  遍历邻接矩阵
    for i in range(ev_dag.shape[0]):
        for j in range(ev_dag.shape[1]):
            if ev_dag[i, j] == 1:
                candidate_edges.append([i, j])
    #  每个边的索引 [i, j] 转换为节点名称 [i2p[i], i2p[j]]。
    if string:
        candidate_edges = [[i2p[e[0]], i2p[e[1]]]
                           for e in candidate_edges]
    # 函数返回包含图中所有边的列表。如果 string 参数为 True，则边的格式为 [A, B]，其中 A 和 B 是节点的名称；否则，边的格式为 [i, j]，其中 i 和 j 是节点的索引。
    return candidate_edges

# 用于重新可视化一系列有向无环图（DAGs）.这个函数遍历多个数据集、数据大小、数据索引、算法和评分的组合，为每个组合加载先验迭代数据，并将其可视化。
def revisualize():
    # : 遍历一个名为 dataset_list 的列表，该列表包含数据集的名称。
    for dataset in dataset_list:
        # 为当前数据集生成一个映射文件路径。
        mapping_path = f"BN_structure/mappings/{dataset}.mapping"
        # 为当前数据集生成一个真实 DAG 文件的路径。
        true_dag_path = f"BN_structure/{dataset}_graph.txt"
        # 使用 NumPy 的 loadtxt 函数加载映射文件，该文件包含节点名称和它们对应的索引。
        mapping = np.loadtxt(mapping_path, dtype=str)
        # 创建一个字典 i2p，将节点索引映射到节点名称。
        i2p = {i: p for i, p in enumerate(mapping)}
        # 加载真实 DAG 的邻接矩阵。
        true_dag = np.loadtxt(true_dag_path, dtype=int)
        # 调用 matrix2edge 函数将邻接矩阵转换为边的列表，使用节点名称。
        true_edges = matrix2edge(true_dag, i2p)
        # 为真实 DAG 生成 PNG 和 SVG 图片的保存路径。
        GT_png_path = f"img/dag_true/{dataset}.png"
        GT_svg_path = f"img/dag_true/{dataset}.svg"
        # 创建一个 GraphVis 类的实例，用于可视化 DAG。
        graph = GraphVis(mapping, true_edges)
        # 初始化 GraphVis 实例的辅助边信息，这里没有存在的边和禁止的边，只有真实边。
        graph.init_aux_edges([], [], true_edges)
        # 调用 visualize 方法来可视化并保存真实 DAG。
        graph.visualize(GT_png_path, GT_svg_path)
        # 遍历一个名为 data_index_list 的列表，它包含数据索引。
        for datasize_index in [0, 1]:
            # 根据数据集和大小索引从 dataset2size 字典中获取数据大小。
            size = dataset2size[datasize_index][dataset]
            # 遍历一个名为 data_index_list 的列表，它包含数据索引。
            for data_index in data_index_list:
                #  遍历一个名为 alg_score_list 的列表，它包含算法和评分的组合。
                for alg, score in alg_score_list:
                    # 生成一个实验名称，用于文件命名。
                    exp_name = f"{dataset}-{size}-{data_index}-{alg}-{score}"
                    # 为当前实验生成一个 JSON 文件的路径，该文件包含先验迭代数据。
                    prior_iter_path = f"out/prior-iter/{exp_name}.json"
                    # 使用 read_json 函数（可能是自定义的或第三方库提供的）加载 JSON 文件。
                    data_prior_iter_raw = read_json(prior_iter_path)
                    #  如果 JSON 文件为空，则跳过当前实验，并打印一条消息。
                    if len(data_prior_iter_raw) == 0:
                        rprint(
                            f"[bold red]Skip[/bold red] [italic blue] {exp_name} [/italic blue] due to no prior iter data")
                        continue
                    #     遍历先验迭代数据。
                    for iter, data in enumerate(data_prior_iter_raw):
                        # 从迭代数据中提取边、存在边和禁止边的信息，并使用 i2p 字典将它们转换为节点名称。
                        edges = [[i2p[e["edge"][0]], i2p[e["edge"][1]]]
                                 for e in data["edges"]]
                        exist_edges = [[i2p[e[0]], i2p[e[1]]]
                                       for e in data["exist_edges"]]
                        forb_edges = [[i2p[e[0]], i2p[e[1]]]
                                      for e in data["forb_edges"]]
                        # 打印实验名称和迭代号。
                        rprint(f"{exp_name}-iter{iter+1}")
                        # 打印存在边和禁止边的信息。
                        rprint(f"exist_edges: {exist_edges}")
                        rprint(f"forb_edges: {forb_edges}")
                        # 为当前迭代生成 PNG 和 SVG 图片的保存路径。
                        png_path = f"img/graph/{exp_name}-iter{iter+1}.png"
                        svg_path = f"img/graph-svg/{exp_name}-iter{iter+1}.svg"
                        #  创建一个新的 GraphVis 实例，用于可视化当前迭代的数据。
                        graph = GraphVis(mapping, edges)
                        # 初始化 GraphVis 实例的辅助边信息。
                        graph.init_aux_edges(
                            exist_edges, forb_edges, true_edges)
                        # 调用 visualize 方法来可视化并保存当前迭代的 DAG。
                        graph.visualize(png_path, svg_path)

# 这段代码设置了脚本的运行环境，定义了一系列的参数，然后调用 revisualize 函数来执行主要的可视化逻辑。
# 脚本的结构是模块化的，使得它可以作为一个独立的脚本运行，也可以作为模块导入到其他脚本中使用。

# 入口
if __name__ == "__main__":
    # 定义一个变量 result_dir，它指定了一个目录，该目录可能用于存储结果或指标。
    result_dir = "out/metrics"
    # 定义一个列表 dataset2size，其中包含了数据集名称到数据大小的映射。
    dataset2size = [
        {"asia": 250, "child": 500, "insurance": 500, "alarm": 1000,
            "cancer": 250, "mildew": 8000, "water": 1000, "barley": 2000},
        {"asia": 1000, "child": 2000, "insurance": 2000, "alarm": 4000,
            "cancer": 1000, "mildew": 32000, "water": 4000, "barley": 8000}
    ]
    # 定义一个列表 data_index_list，它包含了数据索引的集合，可能用于标识不同的数据集实例。
    data_index_list = [1, 2, 3, 4, 5, 6]
    # 定义一个列表 alg_score_list，它包含了算法和评分的组合。每个元素是一个包含两个字符串的列表，第一个字符串是算法名称，第二个字符串是评分方法。
    alg_score_list = [["CaMML", "mml"],  ["HC", "bdeu"], ["softHC", "bdeu"], ["hardMINOBSx", "bdeu"], ["softMINOBSx", "bdeu"],
                      ["HC", "bic"], ["softHC", "bic"], ["hardMINOBSx", "bic"], ["softMINOBSx", "bic"]]
    # 定义一个列表 dataset_list，它包含了要处理的数据集名称。
    dataset_list = ["cancer", "asia", "child",
                    "insurance", "alarm", "mildew", "water", "barley"]
    # 这个函数将遍历数据集、数据大小、数据索引、算法和评分的组合，并重新可视化每个组合的结果。
    revisualize()
