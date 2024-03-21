#!/bin/sh
set -eu

python ResultsCSV.py 1 > tests/out1.tmp 2> /dev/null
python ResultsCSV.py 2 > tests/out2.tmp 2> /dev/null
cat tests/out1.tmp tests/out2.tmp | sort - > tests/sorted.tmp
diff -s tests/sorted.tmp tests/01_reference.csv
