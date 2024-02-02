#!/bin/bash
#SBATCH --job-name=eeg_euclidean_alignment_exp_1_dataset_3_aligment
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-user=a.bruno@ufabc.edu.br
#SBATCH --gres=gpu:1
#SBATCH --exclude=n[1-5,51,52,53,54,55] 
#SBATCH --output=/mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/logs/eeg_euclidean_alignment_exp_1_dataset_3_%j.log
#SBATCH --mail-type=ALL		

module load /mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/cluster/python

python /mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/src/Bruna/script_moabb_exp1.py --config_file=/mnt/beegfs/home/chevallier/bruno/EEG_EA-Domain-Adap/config/config.yaml  --num_exp 'exp_1' --dataset 'Lee2019_MI' --ea 'alignment'


 

