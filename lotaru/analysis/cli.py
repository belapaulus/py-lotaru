import sys
import argparse

from lotaru.analysis.lotaru_scripts import registered_scripts as lotaru_scripts

class Cli:
    def __init__(self):
        self.commands = {
            "list": self.list,
            "describe": self.describe,
            "run": self.run,
            "help": self.help
            }
        self.script_list = sorted(lotaru_scripts, key=lambda x: x.name)
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
        Print the help message for a given analysis script. This is equivalent to
        run <script> --help
        """
        self.run([args[0], "--help"])

    def run(self, args):
        """
        Execute a given analysis script.
        """
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
        
        parser = argparse.ArgumentParser(prog="python -m lotaru run {}".format(script.name),
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
            print(msg)
        else:
            print(self.commands[command[0]].__doc__)


