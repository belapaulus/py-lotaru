#!/bin/env python
import os
import sys
import argparse

from AnalysisScripts import analysis_scripts

class Lotaru2:
    def __init__(self):
        self.commands = {
            "list": self.list,
            "describe": self.describe,
            "run": self.run,
            "help": self.help
            }
        self.analysis_scripts = sorted(analysis_scripts, key=lambda x: x.name)

    def start(self, args):
        if len(args) == 0 or args[0] not in self.commands.keys():
            self.help()
        else:
            self.commands[args[0]](args[1:])

    def list(self, args):
        for i in range(len(self.analysis_scripts)):
            print(i, self.analysis_scripts[i].name)

    def describe(self, args):
        try:
            n = int(args[0])
            analysis_script = self.analysis_scripts[n]
        except (ValueError, IndexError):
            print("invalid experiment number", file=sys.stderr)
            exit(-1)

        print(analysis_script.description)


    def run(self, args):
        try:
            n = int(args[0])
            analysis_script = self.analysis_scripts[n]
        except (ValueError, IndexError):
            print("invalid experiment number", file=sys.stderr)
            exit(-1)
        
        parser = argparse.ArgumentParser(prog="lotaru2 run <script_number>")
        analysis_script.func(parser, args[1:])


    def help(self):
        print("this will print a help message")

if __name__ == "__main__":
    lotaru2 = Lotaru2()
    lotaru2.start(sys.argv[1:])
