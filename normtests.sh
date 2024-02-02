#!/bin/bash

NORM_VALUES=("Identity" "BatchNorm2d")
DROPOUT_VALUES=("drop02")

for NORM1 in "${NORM_VALUES[@]}"; do
    for NORM2 in "${NORM_VALUES[@]}"; do
        for DROPOUT in "${DROPOUT_VALUES[@]}"; do
            COMMAND="python src/Bruna/domain_adapt_exp1.py --model EEGNetNormTest --config_file normtest_configs/config_${DROPOUT}.yaml --ea no-alignment --freeze no-freeze --num_exp exp_shared --dataset BNCI2014001 --sharednorm ${NORM1} --uniquenorm ${NORM2}"

            eval $COMMAND
        done
    done
done