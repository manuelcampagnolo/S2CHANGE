#!/bin/bash

#SBATCH --job-name=GEE              # Nome do Job
#SBATCH --time=48:00:00               # Tempo de execução (HH:MM:SS)
#SBATCH --nodes=1                     # Número de nós
#SBATCH --ntasks-per-node=24          # Número de tarefas por nó 
#SBATCH --mem-per-cpu=5333            # Memória por CPU (8 GB por CPU)
#SBATCH --partition=fct               # Partição
#SBATCH --account=cpca070342024       # Conta
#SBATCH --qos=cpca070342024           # Qualidade de serviço (QoS)

# Carregar e executar o script Python com srun
stdbuf -oL python /users1/cpca070342024/scaetano/CCD_yml_win/S2CHANGE/scripts/pyccd_theia/notebooks/download-gee-36-parts-portugal.py
