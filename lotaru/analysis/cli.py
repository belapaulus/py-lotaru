import sys
import textwrap
import argparse

from lotaru.analysis.lotaru_scripts import registered_scripts as lotaru_scripts
from lotaru.analysis.trace_scripts import registered_scripts as trace_scripts


class Cli:
    def __init__(self):
        self.commands = {
            "list": self.list,
            "describe": self.describe,
            "run": self.run,
            "help": self.help
        }
        all_scripts = lotaru_scripts + trace_scripts
        self.script_list = sorted(all_scripts, key=lambda x: x.name)
        self.script_dict = {}
        for script in self.script_list:
            if script.name in self.script_dict:
                raise RuntimeError
            self.script_dict[script.name] = script

    def start(self, args):
        """
        Start the cli. Call help() if no command is given or if the command is
        unknown. Otherwise execute the command
        """
        if len(args) == 0 or args[0] not in self.commands.keys():
            self.help([])
        else:
            self.commands[args[0]](args[1:])

    def list(self, args):
        """
        List all available analysis scripts.
        """
        for i in range(len(self.script_list)):
            print(i, self.script_list[i].name)

    def describe(self, args):
        """
        Print the help message for a given analysis script. This is equivalent
        to run <script> --help
        """
        if len(args) == 0:
            msg = """
                Usage: 'python -m lotaru describe (<script_name>|<script_number>)'

                Available scripts can be listed with 'python -m lotaru list'.
            """
            print(textwrap.dedent(msg), file=sys.stderr)
            exit(-1)
        self.run([args[0], "--help"])

    def run(self, args):
        """
        Execute a given analysis script.
        """
        if len(args) == 0:
            msg = """
                Usage: 'python -m lotaru run (<script_name>|<script_number>)'

                Available scripts can be listed with 'python -m lotaru list'.
            """
            print(textwrap.dedent(msg), file=sys.stderr)
            exit(-1)
        s = args[0]
        try:
            if s.isdigit():
                n = int(s)
                script = self.script_list[n]
            else:
                script = self.script_dict[s]
        except IndexError:
            print("invalid experiment number", file=sys.stderr)
            exit(-1)
        except KeyError:
            print("invalid experiment name", file=sys.stderr)
            exit(-1)
        parser = argparse.ArgumentParser(
            prog="python -m lotaru run {}".format(script.name),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=script.description)
        script.func(parser, args[1:])

    def help(self, command):
        """
        Print a help message.
        """
        if len(command) == 0:
            msg = """
            Run analysis scripts.

            Available commands:
            {}

            Run help <command> for more information.
            """.format(", ".join(self.commands.keys()))
            print(textwrap.dedent(msg))
            return

        if command[0] in self.commands:
            print(textwrap.dedent(self.commands[command[0]].__doc__))
            return

        print("unknown command: " + command[0])
