"""
Data structures, functions and decorators to define and register analysis scripts
that can be executed via the cli.

Decorators are used to:
 - declare a function as an analysis script
 - add options to an analysis script
 - register an anlysis script with the cli

For an explanation on how decorators work see:
https://peps.python.org/pep-0318/
"""
import argparse
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

