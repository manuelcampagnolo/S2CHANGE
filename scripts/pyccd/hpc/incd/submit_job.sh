#!/bin/bash
#SBATCH --job-name=PyMPI              # Nome do Job
#SBATCH --time=24:00:00               # Tempo de execução (HH:MM:SS)
#SBATCH --nodes=5                     # Número de nós
#SBATCH --ntasks-per-node=24          # Número de tarefas por nó
#SBATCH --cpus-per-task=4            # CPUs por tarefa
#SBATCH --mem-per-cpu=5333            # Memória por CPU
#SBATCH --partition=fct               # Partição
#SBATCH --account=cpca070342024       # Conta
#SBATCH --qos=cpca070342024           # QoS

# Carregar módulos necessários
module load gcc13/openmpi/4.1.6

echo "Nós alocados: $SLURM_NODELIST"
echo "Tarefas alocadas: $SLURM_NTASKS"
echo "CPUs por tarefa: $SLURM_CPUS_PER_TASK"
echo "CPUs utilizados: $(taskset -pc $$)"

export OMPI_MCA_btl=self,tcp
export OMPI_MCA_orte_base_help_aggregate=0

# Executar o script Python com MPI
#mpiexec -n $SLURM_NTASKS python /users1/cpca070342024/scaetano/S2CHANGE/scripts/pyccd/hpc/main_mpi.py
mpiexec --bind-to core --map-by slot:pe=$SLURM_CPUS_PER_TASK -n $SLURM_NTASKS python /users1/cpca070342024/scaetano/S2CHANGE/scripts/pyccd/hpc/main_mpi.py
