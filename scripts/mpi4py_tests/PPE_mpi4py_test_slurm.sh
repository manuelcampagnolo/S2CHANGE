#!/bin/bash

#SBATCH --job-name=ppe_mpi4py_test
#SBATCH --time=0:10:0
#SBATCH --nodes=5
# ntasks-per-node=10, but caused out of memory error
#SBATCH --ntasks-per-node=1
# requested all cpus when testing
#SBATCH --cpus-per-task=96

# Be sure to request the correct partition to avoid the job to be held in the queue, furthermore
#       on CIRRUS-B (Minho)  choose for example HPC_4_Days
#       on CIRRUS-A (Lisbon) choose for example hpc
#SBATCH --partition=fct

#SBATCH --account=cpca070342024

#SBATCH --qos=cpca070342024

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load software modules. Please check session software for the details
module load gcc13/openmpi/4.1.6
module load python/3.10

source venv/mpi_test_venv/bin/activate

# Prepare
src='PPE_mpi4py_test.py'

# Run application. Please note that the number of cores used by MPI are assigned in the SBATCH directives.
echo "=== Running ==="
if [ -e $src ]; then
    mpiexec -np $SLURM_NTASKS venv/mpi_test_venv/bin/python $src
else
    echo "Error: Source file $src not found"
    exit 1
fi

echo "Finished with job $SLURM_JOBID"