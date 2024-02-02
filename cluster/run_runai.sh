#!/bin/bash
set -ex # Enable 'set -e' (exit on error) and 'set -x' (debugging) options

TAG=eeg_ea
types=("alignment")
# "no_alignment")
#types=("no_alignment" )
datasets=("Schirrmeister2017" "PhysionetMI")
#datasets=("Lee2019_MI")
# "Cho2017") 
for type in "${types[@]}"; do
    for dataset in "${datasets[@]}"; do
	string=eeg-ea-${dataset}-no-alignment
	declare -l string
	string=$string  
	runai submit \
	--name ${string} \
	--image aicregistry:5000/${USER}:${TAG} \
	--backoff-limit 0 \
	--cpu-limit 10  \
	--gpu-memory 2G \
	--large-shm \
	--host-ipc \
	--project wds20 \
	--run-as-user \
	--node-type "A100" \
	--volume /nfs/home/wds20/bruno/project/moabb/dataset/:/project/dataset/ \
	--volume /nfs/home/wds20/bruno/project/eeg-ea/:/project  \
	--command -- /usr/bin/python /project/src/Bruna/script_moabb_exp1.py --config_file=/project/config/config.yaml --num_exp 'exp_1' --dataset ${dataset} --ea ${type}
    done
done
