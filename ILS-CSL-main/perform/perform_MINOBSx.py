
import os
import re

import numpy as np

from utils import parse_parents_score, soft_constraint, write_parents_score

# 解析MINOBSx算法屏幕输出，并提取出DAG结构。
# file_path: 字符串，表示MINOBSx算法屏幕输出的文件路径。
# n: 整数，表示BN中变量的数量。
def parse_screen_output(file_path, n):
    # 创建一个大小为n x n的邻接矩阵res，初始值为0。
    # 打开指定路径的文件，并逐行读取。
    # 使用正则表达式匹配每行中的子表达式，提取出子节点和父节点信息。
    # 将提取出的父节点信息转换为整数，并更新邻接矩阵res。
    # 返回解析出的DAG邻接矩阵。
    res = np.zeros((n, n))
    with open(file_path, "r", encoding="utf-8") as ifh:
        for line in ifh:
            mt = re.match(r".*=\s(\d+).*\{(.*)\}.*", line)
            if not mt:
                continue
            child = mt.group(1)
            parents = mt.group(2)
            parents = re.split(r"\s+", parents.strip())
            c = int(child)
            for p in parents:
                if p == "":
                    continue
                p = int(p)
                res[p, c] = 1
    # 返回值：一个numpy数组res，表示DAG的邻接矩阵。
    return res

# 执行MINOBSx算法，解析其输出，并返回DAG结构。
# MINOBSx_base: 字符串，表示MINOBSx算法的根目录。
# timeout: 整数，表示MINOBSx算法运行的超时时间。
# iter: 整数，表示MINOBSx算法迭代的次数。
# prefix: 字符串，用于指定输出文件的前缀。
# prior_confidence: 浮点数，表示先验信息的置信度。
# **kwargs: 可变关键字参数，用于传递其他参数，如数据集名称（d）、样本集名称（s）、关系集名称（r）、存在的边（exist_edges）、禁止的边（forb_edges）等。
def seperate_MINOBSx_unit(MINOBSx_base="minobsx", timeout=None, iter=10, prefix="", prior_confidence=0.99999, **kwargs):
    # 更改当前工作目录到MINOBSx算法的根目录。
    # 创建一个文件anc_path，其中包含先验信息，如存在的边、禁止的边、顺序等。
    # 创建一个文件out_path，用于存储MINOBSx算法的输出。
    # 设置score_filepath参数，用于指定评分函数的路径。
    # 根据is_soft参数的值，创建一个文件anc_path，并运行MINOBSx算法。
    # 解析MINOBSx算法的输出，并返回DAG结构。
    # 清理临时文件和目录。
    os.chdir(MINOBSx_base)

    anc_path = f"anc_file/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.anc"
    out_path = f"out_BNs/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.dne"
    
    score_filepath = kwargs["score_filepath"]
    
    if kwargs["is_soft"] == False:
        with open(anc_path, "w", encoding="utf-8") as ofh:
            ofh.write(f"{len(kwargs['exist_edges'])}\n")
            for c in kwargs['exist_edges']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"0\n")  # undirected edges
            ofh.write(f"{len(kwargs['forb_edges'])}\n")
            for c in kwargs['forb_edges']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"{len(kwargs['order'])}\n")
            for c in kwargs['order']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"{len(kwargs['ancs'])}\n")
            for c in kwargs['ancs']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"{len(kwargs['forb_ancs'])}\n")
            for c in kwargs['forb_ancs']:
                ofh.write(f"{c[0]} {c[1]}\n")
        if timeout is None:
            os.system(
                f"./run-one-case.sh ../{score_filepath} {anc_path} tmp.output {iter} > {out_path}")
        else:
            os.system(
                f"timeout {timeout} ./run-one-case.sh ../{score_filepath} {anc_path} tmp.output {iter} > {out_path}")
        ev_dag = parse_screen_output(f"{out_path}", kwargs['true_dag'].shape[0])
        os.system(f"rm {anc_path}")
        os.system(f"rm {out_path}")
        os.chdir("..")
        return ev_dag
    else:
        with open(anc_path, "w", encoding="utf-8") as ofh:
            ofh.write(f"0\n0\n0\n0\n0\n0\n")
        soft_score_path = f"score/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.score"
        score_dict = parse_parents_score(f"../{score_filepath}")
        edge_cons, forb_cons = kwargs["exist_edges"], kwargs['forb_edges']
        soft_scorer = soft_constraint(obligatory=edge_cons, forbidden=forb_cons, lamdba=[[prior_confidence for _ in range(len(edge_cons))],[prior_confidence for _ in range(len(forb_cons))]])
        for var in score_dict:
            for i, ls in enumerate(score_dict[var]):
                score, parent_set = ls
                prior_bonus = soft_scorer.calculate(var=int(var),parent=[int(p) for p in parent_set])
                new_score = score + prior_bonus
                score_dict[var][i] = (new_score, parent_set)
        write_parents_score(score_dict=score_dict, file_path=soft_score_path)
        if timeout is None:
            os.system(
                f"./run-one-case.sh {soft_score_path} {anc_path} tmp.output {iter} > {out_path}")
        else:
            os.system(
                f"timeout {timeout} ./run-one-case.sh {soft_score_path} {anc_path} tmp.output {iter} > {out_path}")
        ev_dag = parse_screen_output(f"{out_path}", kwargs['true_dag'].shape[0])
        os.system(f"rm {anc_path}")
        os.system(f"rm {out_path}")
        os.system(f"rm {soft_score_path}")
        os.chdir("..")
        # 返回值：一个numpy数组ev_dag，表示执行MINOBSx算法后得到的DAG结构。
        return ev_dag