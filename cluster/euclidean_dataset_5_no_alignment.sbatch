#!/bin/bash
#SBATCH --job-name=eeg_euclidean_alignment_exp_1_dataset_5_no_alignment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-user=a.bruno@ufabc.edu.br
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/logs/eeg_euclidean_dataset_5_no_alignment_%j.log
#SBATCH --mail-type=ALL		

module load /mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/cluster/python

python /mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/src/Bruna/script_moabb_exp1.py --config_file=/mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/config/config.yaml  --num_exp 'exp_1' --dataset 'PhysionetMI' --ea 'no_alignment'


 

