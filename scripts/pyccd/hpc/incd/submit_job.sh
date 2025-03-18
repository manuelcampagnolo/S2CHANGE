#!/bin/bash
#SBATCH --job-name=PyMPI              # Job Name
#SBATCH --time=24:00:00               # Execution Time (HH:MM:SS)
#SBATCH --nodes=5                     # Number of Nodes
#SBATCH --ntasks-per-node=96          # Number of Tasks per Node
#SBATCH --cpus-per-task=1            # CPUs per Task
#SBATCH --mem-per-cpu=5333            # Memory per CPU
#SBATCH --partition=fct               # Partition
#SBATCH --account=cpca070342024       # Account
#SBATCH --qos=cpca070342024           # QoS

# Load necessary modules
module load gcc13/openmpi/4.1.6

echo "Allocated Nodes: $SLURM_NODELIST"
echo "Allocated Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "CPUs in Use: $(taskset -pc $$)"

export OMPI_MCA_btl=self,tcp
export OMPI_MCA_orte_base_help_aggregate=0

# Run the Python script with MPI
mpiexec -n $SLURM_NTASKS python /users1/cpca070342024/scaetano/S2CHANGE/scripts/pyccd/hpc/main_mpi.py
