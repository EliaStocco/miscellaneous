#!/bin/bash -l
# Standard output and error:
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J nvt
## Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#
#SBATCH --mail-type=END
#SBATCH --mail-user=stocco@fhi-berlin.mpg.de
#
# Wall clock limit:
#SBATCH --time=08:00:00

# TEMPLATE for running i-pi.
# 12 October 2020

###################################################################
rm slurm/*
rm files/pid.txt
exec >> slurm/my_output.txt 2>&1   # Redirect all subsequent output to output.txt
source run.sh $@
