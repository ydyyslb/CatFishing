import argparse
import pickle

from pygobnilp.gobnilp import *

# 保存用于计算分数的代码，例如 BIC 和 BDeu。
class myGobnilp(Gobnilp):
    # Added a function, only counting points, no follow-up operations
    # 返回值：本地分数。
    # 内部逻辑：
    # 检查参数的有效性，包括数据源、评分函数、数据类型等。
    # 根据阶段设置数据和本地分数。
    # 如果从数据开始学习，则读取数据并提取约束。
    # 如果从本地分数开始学习，则直接使用提供的本地分数。
    # 计算或加载本地分数。
    # 将本地分数输入模型。
    # 如果设置了输出分数，则将其写入文件。
    def calculate_parent_score(self, data_source=None, varnames=None,
                               header=True, comments='#', delimiter=None,
                               start='no data', end='output written', data_type='discrete',
                               score='BDeu', local_score_fun=None,
                               k=1, ls=False, standardise=False,
                               arities=None, palim=3,
                               alpha=1.0, nu=None, alpha_mu=1.0, alpha_omega=None,
                               starts=(), local_scores_source=None,
                               nsols=1, kbest=False, mec=False, consfile=None, settingsfile=None, pruning=True, edge_penalty=0.0, plot=True,
                               abbrev=True, output_scores=None, output_stem=None, output_dag=True, output_cpdag=True, output_ext=("pdf",),
                               verbose=0, gurobi_output=False, **params):
        # 参数：
        # data_source: 数据源，可以是文件路径或数组。
        # varnames: 变量名称。
        # header: 是否包含头信息。
        # comments: 注释字符串。
        # delimiter: 数据分隔符。
        # start: 学习开始阶段。
        # end: 学习结束阶段。
        # data_type: 数据类型。
        # score: 用于计算局部分数的评分函数名称。
        # local_score_fun: 局部分数函数。
        # k: 评分函数的惩罚乘数。
        # ls: 对于高斯评分，是否使用-MSE/2。
        # standardise: 是否标准化连续数据。
        # arities: 离散变量的基数。
        # palim: 父集的最大大小。
        # alpha: BDeu局部分数生成的等效样本大小。
        # nu: BGe评分中正态部分的均值向量。
        # alpha_mu: BGe评分中正态部分的虚样本大小值。
        # alpha_omega: BGe评分中Wishart部分的自由度。
        # starts: 初始可行DAGs的序列。
        # local_scores_source: 本地分数源。
        # nsols: 学习的BN数量。
        # kbest: 是否将nsols个学习的BN作为一个最高得分的nsols个BN的集合。
        # mec: 是否只对每个Markov等价类学习一个BN。
        # consfile: 包含用户约束的文件。
        # settingsfile: 包含方法参数的文件。
        # pruning: 是否不包括在只有无环性约束时不可能最优的父集。
        # edge_penalty: 父集的边惩罚。
        # plot: 是否绘制学习到的BN/CPDAG。
        # abbrev: 绘图时是否缩写变量名。
        # output_scores: 输出本地分数的文件路径。
        # output_stem: 学习到的BN的输出文件名。
        # output_dag: 是否将DAG写入输出文件。
        # output_cpdag: 是否将CPDAG写入输出文件。
        # output_ext: 文件扩展名。
        # verbose: 添加变量和约束时显示的信息量。
        # gurobi_output: 是否显示Gurobi生成的输出。
        # **params: 任意Gurobi模型参数设置。

        '''
        Args:
            data_source (str/array_like) : If not None, name of the file containing the discrete data or an array_like object.
                                If None, then it is assumed that  data has previously been read in.
            varnames (iterable/None): Names for the variables in the data. If `data_source` is a filename then
                                this value is ignored and the variable names are those given in the file.
                                Otherwise if None then the variable names will X1, X2, ...
            header (bool) : Ignored if `data` is not a filename with continuous data.
                                Whether a header containing variable names is the first non-comment line in the file.
            comments (str) : Ignored if `data` is not a filename with continuous data. Lines starting with this string are treated as comments.
            delimiter (None/str) : Ignored if `data` is not a filename with continuous data.
                                String used to separate values. If None then whitespace is used.
            start (str): Starting stage for learning. Possible stages are: 'no data', 'data', 'local scores',
            'MIP model', 'MIP solution', 'BN(s)' and 'CPDAG(s)'.
            end (str): End stage for learning. Possible values are the same as for `start`.
            data_type (str): Indicates the type of data. Must be either 'discrete' or 'continuous'
            score (str): Name of scoring function used for computing local scores. Must be one of the following:
            BDeu, BGe,
            DiscreteLL, DiscreteBIC, DiscreteAIC,
            GaussianLL, GaussianBIC, GaussianAIC, GaussianL0. This value is ignored if `local_score_fun` is not None.
            local_score_fun (fun/None): If not None a local score function such that `local_score_fun(child,parents)`
                computes `(score,ub)` where `score` is the desired local score for `child` having parentset `parents`
                and `ub` is either `None` or an upper bound on the local score for `child` with any proper superset of `parents`
            k (float): Penalty multiplier for penalised log-likelihood scores (eg BIC, AIC) or tuning parameter ('lambda^2) for l_0
                    penalised Gaussian scoring (as per van de Geer and Buehlmann)
            ls (bool): For Gaussian scores, whether the unpenalised score should be -(1/2) * MSE, rather than log-likelihood
            standardise (bool) : Whether to standardise continuous data.
            arities (array_like/None): Arities for the discrete variables. If `data_source` is a filename then
                                this value is ignored and the arities are those given in the file.
                                Otherwise if None then the arity for a variable is set to the number of distinct
                                values observed for that variable in the data. Ignored for continuous data.
            palim (int/None): If an integer, this should be the maximum size of parent sets.
            alpha (float): The equivalent sample size for BDeu local score generation.
            nu (iter/None): The mean vector for the Normal part of the normal-Wishart prior for BGe scoring.
                            If None then the sample mean is used.
            alpha_mu (float): Imaginary sample size value for the Normal part of the normal-Wishart prior for BGe scoring.
            alpha_omega (float/None): Degrees of freedom for the Wishart part of the normal-Wishart prior for BGe scoring.
                        Must be at least the number of variables. If None then set to 2 more than the number of variables.
            starts (iter): A sequence of feasible DAGs the highest scoring one of which will be the initial
            incumbent solution. Each element in the sequence can be either a bnlearn model string or an nx.DiGraph instance.
            If this value is not empty, a local scoring function must be provided.
            local_scores_source (str/file/dict/None): Ignored if None. If not None then local scores are not computed from data. 
            but come from `local_scores_source`. If a string then the name of a file containing local scores. 
            If a file then the file containing local scores. 
            If a dictionary, then ``local_scores[child][parentset]`` is the score for ``child`` having parents ``parentset``
            where ``parentset`` is a frozenset.
            nsols (int): Number of BNs to learn
            kbest (bool): Whether the `nsols` learned BNs should be a highest scoring set of `nsols` BNs.
            mec (bool): Whether only one BN per Markov equivalence class should be feasible.
            consfile (str/None): If not None then a file (Python module) containing user constraints. 
            Each such constraint is stored indefinitely and it is not possible to remove them.
            settingsfile (str/None): If not None then a file (Python module) containing values for the arguments for this method.
            Any such values override both default values and any values set by the method caller.
            pruning(bool): Whether not to include parent sets which cannot be optimal when acyclicity is the only constraint.
            edge_penalty(float): The local score for a parent set with `p` parents will be reduced by `p*edge_penalty`.
            plot (bool): Whether to plot learned BNs/CPDAGs once they have been learned.
            abbrev (bool): When plotting whether to abbreviate variable names to the first 3 characters.
            output_scores (str/file/None): If not None, then a file or name of a file to write local scores
            output_stem (str/None): If not None, then learned BNs will be written to "output_stem.ext" for each extension defined in 
            `output_ext`. If multiple DAGs have been learned then output files are called "output_stem_0.ext",
            "output_stem_1.ext" ...
            output_dag (bool): Whether to write DAGs to any output files
            output_cpdag (bool): Whether to write CPDAGs to any output files
            output_ext (tuple): File extensions.
            verbose (int) : How much information to show when adding variables and constraints (and computing scores)
            gurobi_output (bool) : Whether to show output generated by Gurobi.
            **params : Arbitrary Gurobi model parameter settings. For example if this method is called with TimeLimit=3, then
            the Gurobi model parameter TimeLimit will be set to 3

        Raises:
            ValueError: If `start= 'no data'` but no data source or local scores source has been provided 
            '''
        if settingsfile is not None:
            if settingsfile.endswith(".py"):
                settingsfile = settingsfile[:-3]
            setmod = importlib.import_module(settingsfile)
            argdkt = {}
            _local = locals()
            for arg in inspect.getfullargspec(Gobnilp.learn).args:
                if arg != 'self':
                    argdkt[arg] = getattr(setmod, arg, _local[arg])
            argdkt['settingsfile'] = None
            return self.learn(**argdkt)

        # if called from R palim will be a float so this needs correcting
        if palim is not None:
            palim = int(palim)

        for stage, stage_str in [(start, 'Starting'), (end, 'End')]:
            if stage not in self.stages_set:
                raise ValueError(
                    "{0} stage '{1}' not recognised.".format(stage_str, stage))
        if not self.before(start, end):
            raise ValueError("Starting stage must come before end stage.")

        if self.before(self._stage, start):
            raise ValueError(
                "Current stage is {0}, but trying to start from later stage {1}".format(self._stage, start))
        else:
            # OK, to perhaps rewind
            self._stage = start

        if score not in self._known_local_scores:
            raise ValueError(
                "Unrecognised scoring function: {0}".format(score))

        if data_type != 'discrete' and data_type != 'continuous':
            raise ValueError(
                "Unrecognised data type: {0}. Should be either 'discrete' or 'continuous'".format(data_type))

        if data_source is not None and local_scores_source is not None:
            raise ValueError("Data source {0} and local scores source {1} both specified. Should specify only one.".format(
                data_source, local_scores_source))

        self._verbose = verbose

        self.Params.OutputFlag = gurobi_output

        for k, v in params.items():
            self.setParam(k, v)

        user_conss_read = False

        if self.between(self._stage, 'data', end):
            if data_source is None and local_scores_source is None:
                raise ValueError(
                    "Learning starting state is 'no data', but no data source or local scores source has been specified.")
            if local_scores_source is None:
                # no data yet, so read it in
                if data_type == 'discrete':
                    self._data = DiscreteData(
                        data_source, varnames=varnames, arities=arities)
                elif data_type == 'continuous':
                    self._data = ContinuousData(data_source, varnames=varnames, header=header,
                                                comments=comments, delimiter=delimiter, standardise=standardise)

                # BN variables always in order
                self._bn_variables = sorted(self._data.variables())

                # now BN variables have been set can pull in constraints from consfile
                self.input_user_conss(consfile)
                user_conss_read = True

            self._stage = 'data'

        if self.between(self._stage, 'local scores', end):
            # no local scores yet, so compute them ...
            if local_scores_source is None:
                if score == 'BDeu':
                    local_score_fun = BDeu(self._data, alpha=alpha).bdeu_score
                elif score == 'BGe':
                    local_score_fun = BGe(
                        self._data, nu=nu, alpha_mu=alpha_mu, alpha_omega=alpha_omega).bge_score
                else:
                    klass = globals()[score]
                    if score.startswith('Gaussian'):
                        local_score_fun = klass(self._data, k=k, ls=ls).score
                    else:
                        local_score_fun = klass(self._data).score

                # take any non-zero edge penalty into account
                if edge_penalty != 0.0:
                    def local_score_edge(child, parents):
                        score, ub = local_score_fun(child, parents)
                        pasize = len(parents)
                        if ub is not None:
                            ub -= edge_penalty * (pasize+1)
                        return score - edge_penalty * pasize, ub
                    local_score_fun = local_score_edge

                # store any initial feasible solutions now so that relevant scores are computed
                self.set_starts(starts)
                local_scores = self.return_local_scores(
                    local_score_fun, palim=palim, pruning=pruning)

            # ... or read them in
            else:
                if type(local_scores_source) == dict:
                    local_scores = local_scores_source
                else:
                    local_scores = read_local_scores(local_scores_source)

                # remove parent sets with too many parents
                # _enforce_palim(local_scores, palim)

                # apply edge penalty if there is one
                if edge_penalty != 0.0:
                    for child, scoredparentsets in local_scores.items():
                        for parentset, skore in scoredparentsets.items():
                            scoredparentsets[parentset] = skore - \
                                (edge_penalty * len(parentset))

            self.input_local_scores(local_scores)
            if not user_conss_read:  # won't have been read in yet if learning from scores rather than data
                self.input_user_conss(consfile)
                user_conss_read = True
            # self._best_subip()
            if output_scores is not None:
                self.write_local_scores(output_scores)
            self._stage = 'local scores'
        return local_scores

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Use Gurobi for Bayesian network learning',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 添加参数:
# --data_source: 指定包含数据或局部得分的文件。
# help: 描述了这个参数的作用。
# default: 指定了这个参数的默认值。
parser.add_argument("--data_source", help="File containing data or local scores",
                    default='data/mildew_8000_1.txt')
# --output: 指定输出文件的位置。
# type，指定了参数的类型，,这里是 str
parser.add_argument("--output", type=str, default='tmpresult/tmp.txt')
parser.add_argument("--forb_edges", type=str, default=None)
parser.add_argument("--exist_edges", type=str, default=None)

parser.add_argument("--noheader", action="store_true",
                    help="For continuous data only: The first non-comment line in the input file does not list the variables.")
parser.add_argument("--comments", default='#',
                    help="For continuous data only: Lines starting with this string are treated as comments.")
parser.add_argument("--delimiter", action="store_true", default=None,
                    help="For continuous data only: String used to separate values. If not set then whitespace is used.")
parser.add_argument("--end", default="output written",
                    help="End stage for learning. If set to 'local scores' execution stops once local scores are computed")
parser.add_argument("--score", default="DiscreteBIC",
                    help="""Name of scoring function used for computing local scores. Must be one
                    of the following: BDeu, BGe, DiscreteLL,
                    DiscreteBIC, DiscreteAIC, GaussianLL, GaussianBIC,
                    GaussianAIC, GaussianL0.""")
parser.add_argument("--k", default=1,
                    help="""Penalty multiplier for penalised log-likelihood scores (eg BIC, AIC) or tuning parameter ('lambda^2) for l_0
                    penalised Gaussian scoring (as per van de Geer and Buehlmann)""")
parser.add_argument("--ls", action="store_true",
                    help="For Gaussian scores, make unpenalised score -(1/2) * MSE, rather than log-likelihood")
parser.add_argument("--standardise", action="store_true",
                    help="Standardise continuous data.")
parser.add_argument("--palim", "-p", type=int, default=3,
                    help="Maximum size of parent sets.")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="The equivalent sample size for BDeu local score generation.")
parser.add_argument("--alpha_mu", type=float, default=1.0,
                    help="Imaginary sample size value for the Normal part of the normal-Wishart prior for BGe scoring.")
parser.add_argument("--alpha_omega", type=int, default=None,
                    help="""Degrees of freedom for the Wishart part of the normal-Wishart prior for BGe scoring. 
                    Must be at least the number of variables. If not supplied 2 more than the number of variables is used.""")
parser.add_argument('--scores', '-s', action="store_true",
                    help="The input consists of pre-computed local scores (not data)")
parser.add_argument("--nsols", '-n', type=int, default=1,
                    help="Number of BNs to learn")
parser.add_argument("--kbest", action="store_true",
                    help="Whether the nsols learned BNs should be a highest scoring set of nsols BNs.")
parser.add_argument("--mec", action="store_true",
                    help="Make only one BN per Markov equivalence class feasible.")
parser.add_argument("--consfile",
                    help="A file (Python module) containing user constraints.")
parser.add_argument("--settingsfile",
                    help="""A file (Python module) containing values for the arguments for Gobnilp's 'learn' method
                    Any such values override both default values and any values set on the command line.""")
parser.add_argument("--nopruning", action="store_true",
                    help="No pruning of provably sub-optimal parent sets.")
parser.add_argument("--edge_penalty", type=float, default=0.0,
                    help="The local score for a parent set with p parents will be reduced by p*edge_penalty.")
parser.add_argument("--noplot", action="store_true",
                    help="Prevent learned BNs/CPDAGs being plotted.")
parser.add_argument("--noabbrev", action="store_true",
                    help="When plotting DO NOT to abbreviate variable names to the first 3 characters.")
parser.add_argument("--output_scores",
                    help="Name of a file to write local scores")
parser.add_argument("--output_stem", "-o",
                    help="""Learned BNs will be written to 'output_stem.ext' for each extension defined by 
                    `output_ext`. If multiple DAGs have been learned then output files are called 'output_stem_0.ext',
                    'output_stem_1.ext' ... No DAGs are written if this is not set.""")
parser.add_argument("--nooutput_dag", action="store_true",
                    help="Do not write DAGs to any output files")
parser.add_argument("--nooutput_cpdag", action="store_true",
                    help="Do not write CPDAGs to any output files")
parser.add_argument("--output_ext", default='pdf',
                    help="Comma separated file extensions which determine the format of any output DAGs or CPDAGs.")
parser.add_argument("--verbose", '-v', type=int, default=0,
                    help="How much information to show when adding variables and constraints (and computing scores)")
parser.add_argument("--gurobi_output", "-g", action="store_true",
                    help="Whether to show output generated by Gurobi.")
parser.add_argument("--gurobi_params", nargs='+',
                    help="Gurobi parameter settings.")
# 解析命令行参数：
# 这行代码读取命令行参数，并将其存储在 args 变量中。
args = parser.parse_args()
# 转换为字典:
# 这行代码将 args 对象转换为一个字典，方便后续的访问和使用。
argdkt = vars(args)

# process options which set 'learn' arguments to false
# 这段代码的作用是处理那些通过命令行参数设置为 False 的选项。
# 它遍历一系列的选项，如 'noheader', 'nopruning', 'noplot', 'noabbrev', 'nooutput_dag', 和 'nooutput_cpdag'，
# 并将它们的值从 False 更改为 True，同时从 argdkt 字典中删除这些选项。
for opt in 'noheader', 'nopruning', 'noplot', 'noabbrev', 'nooutput_dag', 'nooutput_cpdag':
    argdkt[opt[2:]] = not argdkt[opt]
    del argdkt[opt]

# interpret first argument as local scores file if --scores is used
# 这段代码的作用是处理 --scores 选项，如果该选项被使用，它会将 data_source 选项的值重定向到 local_scores_source 选项，并从 argdkt 字典中删除 --scores 选项。
if argdkt['scores']:
    argdkt['local_scores_source'] = argdkt['data_source']
    del argdkt['data_source']
del argdkt['scores']

# convert string specifying list of extensions to a list
argdkt['output_ext'] = argdkt['output_ext'].split(',')

# assume data is continuous if a score for continuous data is specified
s = argdkt['score']
if s == "BGe" or s.startswith('Gaussian'):
    argdkt['data_type'] = 'continuous'

if argdkt['gurobi_params'] is not None:
    for argval in argdkt['gurobi_params']:
        a, _, v = argval.partition('=')
        if '.' in v:
            argdkt[a] = float(v)
        else:
            try:
                argdkt[a] = int(v)
            except ValueError:
                argdkt[a] = v
del argdkt['gurobi_params']


# do learning
# 创建 GoBN 对象，这个方法实现了贝叶斯网络学习的算法。
GoBN = myGobnilp()
# 处理存在的边
# 如果 argdkt 字典中存在 --exist_edges 选项，并且该选项的值不是 None，则代码会打开指定的文件，使用 pickle.load(f) 函数加载文件中的内容，并将其存储在 exist_edges 变量中。
# 然后，对于文件中的每一对父节点和子节点，代码会调用 GoBN.add_obligatory_arrow(str(i), str(j)) 方法，向 GoBN 对象添加一个强制边。
if argdkt['exist_edges'] != None:
    with open(argdkt['exist_edges'], 'rb') as f:
        exist_edges = pickle.load(f)
    for i, j in exist_edges:
        GoBN.add_obligatory_arrow(str(i), str(j))
# 处理禁止的边
# 如果 argdkt 字典中存在 --forb_edges 选项，并且该选项的值不是 None，则代码会打开指定的文件，使用 pickle.load(f) 函数加载文件中的内容，并将其存储在 forb_edges 变量中。
# 然后，对于文件中的每一对父节点和子节点，代码会调用 GoBN.add_forbidden_arrow(str(i), str(j)) 方法，向 GoBN 对象添加一个禁止边。
if argdkt['forb_edges'] != None:
    with open(argdkt['forb_edges'], 'rb') as f:
        forb_edges = pickle.load(f)
    for i, j in forb_edges:
        GoBN.add_forbidden_arrow(str(i), str(j))
# 计算父节点得分
parent_score = GoBN.calculate_parent_score(**argdkt)
# 写入输出文件:
# 这行代码使用 with open 语句打开 argdkt 字典中 --output 选项指定的文件，并将其写入模式为 ‘w’（写入）。
# 然后，它遍历 parent_score 字典中的每个变量，并为每个变量遍历其父节点。
# 对于每个变量和其父节点，代码创建了一个字符串 tmp，其中包含了父节点的列表。
# 最后，它将变量名、父节点列表和得分写入文件中。
with open(argdkt['output'], 'w') as f:
    for var in parent_score:
        for parent in parent_score[var]:
            tmp = '{'+' '.join(parent)+'}'
            f.write(f'{var} {tmp} {parent_score[var][parent]}\n')
