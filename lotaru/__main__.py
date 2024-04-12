import sys
import warnings

from lotaru.analysis.cli import Cli

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Cli().start(sys.argv[1:])
