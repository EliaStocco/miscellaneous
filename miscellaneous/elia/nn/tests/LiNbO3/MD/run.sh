#!/bin/bash -l

save_suffix=""
hostname="hello"
calculation="pw"
run_ipi='true'
run_ipi_somewhereelse='false'
write_qe='true'
run_qe='true'
run_aims='false'
max_restart="0" # a negative number means infinite
#PARA_PREFIX="mpirun -np 1"
force_start="true"
# ipi_input="input.xml" # comment it or specify an existing file
# OUTPUT_FILE="test.out"
aims_radix='aims'
ipi_radix='i-pi'
nk="1"
############################################################################
# read input arguments
source ~/run.sh $@
