#!/bin/bash
set -ex # Enable 'set -e' (exit on error) and 'set -x' (debugging) options

TAG=eeg_ea
types=("alignment")
# "no-alignment" )
datasets=("Cho2017" "Schirrmeister2017" "PhysionetMI")
datasets=('BNCI2014001')

sessions=('session_T' 'session_E' 'both')
for session in "${sessions[@]}"; do
for type in "${types[@]}"; do
    for dataset in "${datasets[@]}"; do
    	for online in {0..0}; do
	string=exp3-ea-${dataset}-${type}-${session}-${online}
	declare -l string
        string=${string//_/} # removes all underscores from the string
	runai submit \
	--name ${string} \
	--image aicregistry:5000/${USER}:${TAG} \
	--backoff-limit 0 \
	--cpu-limit 10  \
	--gpu-memory 4G \
	--large-shm \
	--host-ipc \
	--project wds20 \
	--run-as-user \
	--node-type "A100" \
	--volume /nfs/home/wds20/bruno/project/moabb/dataset/:/project/dataset/ \
	--volume /nfs/home/wds20/bruno/project/eeg-ea/:/project  \
	--command -- /usr/bin/python /project/src/Bruna/script_moabb_exp3.py --config_file=/project/config/config.yaml --num_exp 'exp_3' --dataset ${dataset} --ea ${type} --online ${online} --session ${session}
	
    done
   done
 done
 sleep 1s
done
