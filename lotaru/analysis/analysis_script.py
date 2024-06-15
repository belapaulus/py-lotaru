from lotaru.RunExperiment import run_experiment
from lotaru.Constants import LOTARU_A_BENCH, LOTARU_G_BENCH
"""
Data structures, functions and decorators to define and register analysis
scripts that can be executed via the cli.

Decorators are used to:
 - declare a function as an analysis script
 - add options to an analysis script
 - register an anlysis script with the cli

For an explanation on how decorators work see:
https://peps.python.org/pep-0318/
"""
from functools import wraps


class AnalysisScript:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func


def register(func_list):
    def setup(func):
        func_list.append(AnalysisScript(func.__name__, func.__doc__, func))
        return func
    return setup


def option(*args, **kwargs):
    def setup(func):
        @wraps(func)
        def func_to_return(arg_parser, arg_string):
            arg_parser.add_argument(*args, **kwargs)
            func(arg_parser, arg_string)
        return func_to_return
    return setup


def defaultanalysis(func):
    @wraps(func)
    def func_to_return(arg_parser, arg_string):
        arg_parser.add_argument("--scale-bayesian-model",
                                type=toBool, default=True)
        arg_parser.add_argument("--scale-median-model",
                                type=toBool, default=False)
        arg_parser.add_argument("-e", "--experiment-number", default="1")
        arg_parser.add_argument('--scaler', choices=['a', 'g'], default='g')
        arg_parser.add_argument('-x', '--resource-x',
                                default="taskinputsizeuncompressed")
        arg_parser.add_argument('-y', '--resource-y', default="realtime")
        args = arg_parser.parse_args(arg_string)
        scaler_bench_file = {
            'a': LOTARU_A_BENCH,
            'g': LOTARU_G_BENCH,
        }
        r = run_experiment(experiment_number=args.experiment_number,
                           resource_x=args.resource_x,
                           resource_y=args.resource_y,
                           scaler_type=args.scaler,
                           scaler_bench_file=scaler_bench_file[args.scaler],
                           scale_median_model=args.scale_median_model,
                           scale_bayesian_model=args.scale_bayesian_model)
        func(args, r)
    return func_to_return


def analysis(func):
    @wraps(func)
    def func_to_return(arg_parser, arg_string):
        args = arg_parser.parse_args(arg_string)
        func(args)
    return func_to_return


def toBool(s):
    """
    helper to parse boolean options
    """
    if not (s == "True" or s == "False"):
        raise RuntimeError
    return s == "True"
