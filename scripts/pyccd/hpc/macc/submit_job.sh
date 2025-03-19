#!/bin/bash
#SBATCH --job-name=PyMPI              # Job Name
#SBATCH --time=4:00:00                # Execution Time (HH:MM:SS)
#SBATCH --nodes=25                     # Number of nodes (64)
#SBATCH --ntasks-per-node=8            # Number of tasks per node (128)
#SBATCH --cpus-per-task=16             # CPUs per task
#SBATCH --partition=normal-x86         # Partition ---> large-x86 number of nodes: 128
#SBATCH --account=F202410004CPCAA1X    # Account

# Load OpenMPI
module load OpenMPI/4.1.5-GCC-12.2.0 || { echo "Error loading OpenMPI"; exit 1; }
export OMPI_MCA_orte_base_help_aggregate=0

# Display Slurm information
echo "Allocated nodes: $SLURM_NODELIST"
echo "Allocated tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Execute the Python script with MPI
mpiexec -n $SLURM_NTASKS python /home/scaetano/S2CHANGE/scripts/pyccd/hpc/main_mpi.py
