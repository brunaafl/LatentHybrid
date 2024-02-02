#!/bin/bash
#SBATCH --job-name=eeg_euclidean_alignment_exp_1_dataset_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-user=a.bruno@ufabc.edu.br 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/logs/eeg_euclidean_exp_1_dataset_1_%j.log		

module load /mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/cluster/python

python /mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/src/Bruna/script_moabb_exp1.py --config_file=/mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/config/config.yaml  --num_exp 'exp_1' --dataset 'BNCI2014001'


 

