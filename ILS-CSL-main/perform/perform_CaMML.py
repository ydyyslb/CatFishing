import os
import re

import numpy as np

# 用于解析 CaMML 算法的输出文件，并将其转换为一个表示有向无环图（DAG）的 NumPy 数组。
# file_path: 字符串，表示 CaMML 输出文件的路径。
# dag_flag: 布尔值，表示是否返回 DAG 数组，默认为 True。
def parse_CaMML_output(file_path, dag_flag=True):
    # node_pattern = r"node\s+(\S+)\s+{.*": 正则表达式模式，用于匹配节点信息。
    # parents_pattern = r".*parents.*\((.*)\).*": 正则表达式模式，用于匹配父节点信息。
    # node_list = []: 创建一个空列表 node_list，用于存储节点的名称。
    # parent_list = []: 创建一个空列表 parent_list，用于存储每个节点的父节点列表。
    node_pattern = r"node\s+(\S+)\s+{.*"
    parents_pattern = r".*parents.*\((.*)\).*"
    node_list = []
    parent_list = []
    with open(file_path, "r", encoding="utf-8") as ifh:
        # 遍历文件中的所有行
        for line in ifh.readlines():
            # 使用正则表达式 node_pattern 匹配节点信息。
            m1 = re.match(node_pattern, line)
            # 如果匹配成功，
            if m1:
                # 提取节点名称，并去除前导和尾随空白字符。
                node = m1.group(1).strip()
                #  将节点名称添加到 node_list 列表中。
                node_list.append(node)
                continue
            #     使用正则表达式 parents_pattern 匹配父节点信息。
            m2 = re.match(parents_pattern, line)
            if m2:
                # 提取父节点列表字符串，并去除前导和尾随空白字符。
                parents = m2.group(1).strip()
                #  如果父节点列表为空，则添加一个空列表到 parent_list 列表中。
                if parents == "":
                    parent_list.append([])
                # 如果父节点列表不为空，tp = parents.split(","): 将父节点列表字符串按逗号分割。
                # tp = [e.strip() for e in tp]: 去除每个父节点的前导和尾随空白字符。
                # parent_list.append(tp): 将处理后的父节点列表添加到 parent_list 列表中。
                else:
                    tp = parents.split(",")
                    tp = [e.strip() for e in tp]
                    parent_list.append(tp)
    # 创建一个空字典 p_dict，用于存储节点的父节点信息。
    # 创建一个空字典 n2i，用于存储节点名称到索引的映射。
    p_dict = {}
    n2i = {}
    for i, n in enumerate(node_list):
        # 在字典 n2i 中为当前节点创建一个键值对，键是节点名称，值是节点索引。
        # 在字典 p_dict 中为当前节点创建一个键值对，键是节点名称，值是父节点列表。
        n2i[n] = i
        p_dict[n] = parent_list[i]

    if not dag_flag:
        return n2i, p_dict
    # 获取节点数量 n
    n = len(n2i)
    CaMML_dag = np.zeros((n, n))
    for v in p_dict:
        # 获取当前节点的索引 v1。
        v1 = n2i[v]
        # 遍历当前节点的父节点列表。
        for p in p_dict[v]:
            # 获取当前父节点的索引 p1。
            p1 = n2i[p]
            # 在 CaMML_dag 数组中，将 (p1, v1) 位置的元素设置为 1，表示存在从父节点 p1 到子节点 v1 的有向边。
            CaMML_dag[p1][v1] = 1
    return CaMML_dag

# CaMML_unit 函数用于执行 CaMML 算法的一个单元操作。它首先创建一个包含祖先约束、禁止祖先约束、禁止边和存在边的字符串，并将其写入一个临时文件。
# 然后，它执行 CaMML 算法，解析输出文件，并返回 DAG 估计结果。最后，它删除临时文件和输出文件，并返回上一级目录。
# prefix: 字符串，表示实验的前缀。
# i2p: 一个字典，将节点索引映射到节点名称。
# **kwargs: 一个关键字参数，包含实验的多个配置选项。
def CaMML_unit(CaMML_base, prefix, i2p, **kwargs):
    # 使用 os.chdir 函数更改当前工作目录到 CaMML_base。
    os.chdir(CaMML_base)

    # 创建一个字符串 anc_path，表示祖先文件路径，其中包含祖先约束和禁止祖先约束的边信息。
    # 创建一个字符串 out_path，表示 CaMML 算法输出的路径，其中包含 DAG 估计结果。
    anc_path = f"anc_file/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.anc"
    out_path = f"out_BNs/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.dne"

    with open(anc_path, "w") as ofh:
        #  计算重构概率 reconf，它是 kwargs['conf'] 的补数。
        reconf = 1-kwargs['conf']
        # 初始化一个字符串 out_str，用于构建输出字符串。
        out_str = "arcs{\n"
        # 遍历祖先约束列表 kwargs['ancs']。
        for a in kwargs['ancs']:
            # 提取祖先约束中的节点索引。
            # 将祖先约束添加到字符串 out_str 中。
            v1, v2 = a
            out_str += f"{i2p[v1]} => {i2p[v2]} {kwargs['conf']};\n"
        #  遍历禁止祖先约束列表 kwargs['forb_ancs']
        for a in kwargs['forb_ancs']:
            # v1, v2 = a: 提取禁止祖先约束中的节点索引。
            # out_str += f"{i2p[v1]} => {i2p[v2]} {reconf};\n": 将禁止祖先约束添加到字符串 out_str 中。
            v1, v2 = a
            out_str += f"{i2p[v1]} => {i2p[v2]} {reconf};\n"
        #     遍历禁止边列表 kwargs['forb_edges']
        for a in kwargs['forb_edges']:
            v1, v2 = a
            out_str += f"{i2p[v1]} -> {i2p[v2]} {reconf:.5f};\n"
        #      遍历存在边列表 kwargs['exist_edges']。
        for a in kwargs['exist_edges']:
            v1, v2 = a
            out_str += f"{i2p[v1]} -> {i2p[v2]} {kwargs['conf']};\n"
        #      添加闭合的 arcs 块。
        out_str += "}"
        # 如果存在顺序约束，
        if len(kwargs['order']) > 0:
            #  添加 tier 块
            out_str += "\ntier{\n"
            # 遍历顺序约束列表
            for order_instance in kwargs['order']:
                # 将顺序约束中的节点名称用 ‘>’ 连接。
                out_str += "<".join([i2p[i] for i in order_instance])
                out_str += ";\n"
            #     添加闭合的 tier 块。
            out_str += "}"
        ofh.write(out_str)

    # 执行 camml.sh 脚本，该脚本用于运行 CaMML 算法。参数包括祖先文件路径 anc_path、数据集路径 ../data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv
    # 和输出路径 out_path。输出被重定向到 log.txt 文件。
    os.system(
        f"./camml.sh -p {anc_path} ../data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv {out_path} > log.txt")
    # 函数解析 CaMML 算法的输出，并将其返回的 DAG 数组存储在变量 ev_dag 中。
    ev_dag = parse_CaMML_output(out_path)

    os.system(f"rm {anc_path}")
    os.system(f"rm {out_path}")
    os.chdir("..")
    # return ev_dag: 函数返回 DAG 估计结果 ev_dag。
    return ev_dag
