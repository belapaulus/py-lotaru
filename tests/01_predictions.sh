#!/bin/sh
set -eu

python -m lotaru run results_csv -o tests/out.tmp
sort tests/out.tmp > tests/sorted.tmp
diff -s tests/sorted.tmp tests/01_reference.csv
rm tests/sorted.tmp tests/out.tmp
