import argparse
import sys

# 用于从命令行加载配置参数。
# 在提供的load_config函数中，以下是默认参数：
#
# --dataset 默认为 "child"
# --model 默认为 "gpt-4"
# --alg 默认为 "PC"
# --score 默认为 "bdeu"
# --data_index 默认为 3
# --log_filepath 默认为 None（这意味着输出将打印到屏幕而不是文件）
# --datasize_index 默认为 1
def load_config():
    """
    Load configuration from command line
    """
    #  创建一个 argparse 对象 parser，并指定其描述。
    parser = argparse.ArgumentParser(
        description='Causal Structure Learning with Iterative LLM queries')
    # 添加一个命令行参数 --dataset，用于指定数据集名称。
    parser.add_argument('--dataset', type=str, default="UCI",
                        help="dataset name, can be one of ['asia', 'alarm', 'child', 'insurance', 'barley', 'mildew', 'cancer', 'water']")
    # 添加一个命令行参数 --model，用于指定使用的语言模型。
    parser.add_argument('--model', type=str, default="gpt-4",
                        help="LLM model, can be one of ['gpt-3.5-turbo', 'gpt-4', 'sim-gpt']")
    # 添加一个命令行参数 --alg，用于指定因果结构学习算法。
    parser.add_argument('--alg', type=str, default="GSP",
                        help="CSL algorithm, can be one of ['CaMML', 'HC', 'softHC', 'hardMINOBSx', 'softMINOBSx','Notears','no_test', 'PC']")
    # 添加一个命令行参数 --score，用于指定评分方法。
    parser.add_argument('--score', type=str, default="bic",
                        help="CSL algorithm, can be one of ['mml', 'bic', 'bdeu']")
    # 添加一个命令行参数 --data_index，用于指定数据索引。
    parser.add_argument('--data_index', type=int, default=1,
                        help="index of data, can be one of [1, 2, 3, 4, 5, 6]")
    #  添加一个命令行参数 --log_filepath，用于指定日志文件路径。
    parser.add_argument('--log_filepath', type=str, default=None,
                        help="log file path, print to screen as default")
    # 添加一个命令行参数 --datasize_index，用于指定数据大小索引。
    parser.add_argument('--datasize_index', type=int, default=0,
                        help="index of datasize params, can be one of [0, 1]")
    # 使用 parser.parse_args() 方法解析命令行参数，并将结果存储在 args 变量中。
    args = parser.parse_args()
    # 如果指定了日志文件路径，
    if args.log_filepath is not None:
        # 将标准输出重定向到指定的日志文件。
        sys.stdout = open('out.txt', 'w')
    # 函数返回包含命令行参数的 argparse 对象。
    return args


if __name__ == '__main__':
    # 调用 load_config 函数来加载配置参数。这个函数在脚本之前定义，用于从命令行解析参数。
    args = load_config()
    # 打印 args 变量，它包含了从命令行解析得到的参数。
    print(args)
