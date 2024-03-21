#!/bin/sh
set -eu

python -c'from Analysis import *; results_csv()' > tests/out.tmp 2> /dev/null
sort tests/out.tmp > tests/sorted.tmp
diff -s tests/sorted.tmp tests/01_reference.csv
