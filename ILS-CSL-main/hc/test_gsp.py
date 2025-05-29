import numpy as np
import pandas as pd
from hc.gsp import gsp, gsp_prior
from hc.DAG import DAG
import matplotlib.pyplot as plt

def test_gsp_vs_hc():
    """
    测试GSP算法与HC算法的对比
    """
    # 生成一些简单的测试数据
    np.random.seed(42)
    n = 100  # 样本数
    d = 5    # 变量数
    
    # 生成DAG: 1->2->3->4->5
    true_dag = np.zeros((d, d))
    for i in range(d-1):
        true_dag[i, i+1] = 1
    
    # 根据DAG生成线性高斯数据
    data = np.zeros((n, d))
    data[:, 0] = np.random.normal(0, 1, n)
    
    for i in range(1, d):
        parents = np.where(true_dag[:, i] == 1)[0]
        noise = np.random.normal(0, 1, n)
        if len(parents) > 0:
            for p in parents:
                data[:, i] += 0.8 * data[:, p] + noise
        else:
            data[:, i] = noise
    
    # 转换为pandas DataFrame
    varnames = [f'X{i}' for i in range(1, d+1)]
    df = pd.DataFrame(data, columns=varnames)
    
    # 使用GSP算法学习DAG
    gsp_dag = gsp(df.to_numpy(), None, varnames, score='bic_g')
    
    # 打印结果
    print("真实DAG:")
    for i in range(d):
        var = varnames[i]
        parents = [varnames[j] for j in range(d) if true_dag[j, i] == 1]
        print(f"{var}: {parents}")
    
    print("\nGSP学习的DAG:")
    for var in gsp_dag:
        print(f"{var}: {gsp_dag[var]['par']}")
    
    # 计算结构汉明距离 (SHD)
    def calculate_shd(true_g, learned_dag, varnames):
        """计算结构汉明距离"""
        learned_g = np.zeros((len(varnames), len(varnames)))
        for i, var in enumerate(varnames):
            for parent in learned_dag[var]['par']:
                learned_g[varnames.index(parent), i] = 1
                
        # SHD计算：不同边的数量
        shd = np.sum(np.abs(true_g - learned_g))
        return shd
    
    shd = calculate_shd(true_dag, gsp_dag, varnames)
    print(f"\n结构汉明距离 (SHD): {shd}")
    
    # 可视化
    def plot_dag(dag, title):
        """绘制DAG"""
        g = np.zeros((len(varnames), len(varnames)))
        for i, var in enumerate(varnames):
            for parent in dag[var]['par']:
                g[varnames.index(parent), i] = 1
                
        plt.figure(figsize=(5, 5))
        plt.imshow(g, cmap='Blues')
        plt.title(title)
        plt.xticks(range(len(varnames)), varnames)
        plt.yticks(range(len(varnames)), varnames)
        for i in range(len(varnames)):
            for j in range(len(varnames)):
                if g[i, j] == 1:
                    plt.text(j, i, '→', ha='center', va='center')
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()
        
    # 将gsp_dag转换为字典形式以便绘制
    plot_dag(gsp_dag, "GSP学习的DAG")
    
    # 生成真实DAG的字典表示
    true_dag_dict = {}
    for i, var in enumerate(varnames):
        true_dag_dict[var] = {'par': []}
        for j in range(len(varnames)):
            if true_dag[j, i] == 1:
                true_dag_dict[var]['par'].append(varnames[j])
                
    plot_dag(true_dag_dict, "真实DAG")


def test_gsp_prior():
    """
    测试带有先验知识的GSP算法
    """
    # 生成测试数据
    np.random.seed(42)
    n = 100  # 样本数
    d = 5    # 变量数
    
    # 生成DAG: 1->2->3->4->5
    true_dag = np.zeros((d, d))
    for i in range(d-1):
        true_dag[i, i+1] = 1
    
    # 根据DAG生成线性高斯数据
    data = np.zeros((n, d))
    data[:, 0] = np.random.normal(0, 1, n)
    
    for i in range(1, d):
        parents = np.where(true_dag[:, i] == 1)[0]
        noise = np.random.normal(0, 1, n)
        if len(parents) > 0:
            for p in parents:
                data[:, i] += 0.8 * data[:, p] + noise
        else:
            data[:, i] = noise
    
    # 转换为pandas DataFrame
    varnames = [f'X{i}' for i in range(1, d+1)]
    df = pd.DataFrame(data, columns=varnames)
    
    # 创建DAG对象
    D = DAG(varnames)
    D.load_data(df)
    
    # 添加一些先验知识
    # 例如，我们知道X1->X2
    D.a_prior['X2']['par'] = ['X1']
    # 我们也禁止X5->X1
    D.d_prior['X1']['par'] = ['X5']
    
    # 使用带先验的GSP算法学习DAG
    gsp_dag = gsp_prior(D)
    
    # 打印结果
    print("\n带先验知识的GSP学习的DAG:")
    for var in gsp_dag:
        print(f"{var}: {gsp_dag[var]['par']}")
    
    # 可视化
    def plot_dag(dag, title):
        """绘制DAG"""
        g = np.zeros((len(varnames), len(varnames)))
        for i, var in enumerate(varnames):
            for parent in dag[var]['par']:
                g[varnames.index(parent), i] = 1
                
        plt.figure(figsize=(5, 5))
        plt.imshow(g, cmap='Blues')
        plt.title(title)
        plt.xticks(range(len(varnames)), varnames)
        plt.yticks(range(len(varnames)), varnames)
        for i in range(len(varnames)):
            for j in range(len(varnames)):
                if g[i, j] == 1:
                    plt.text(j, i, '→', ha='center', va='center')
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()
    
    plot_dag(gsp_dag, "带先验知识的GSP学习的DAG")


if __name__ == "__main__":
    test_gsp_vs_hc()
    test_gsp_prior() 