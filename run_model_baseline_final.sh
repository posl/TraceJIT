#!/bin/bash

use_scale=$1
N=$2
project=$3 
algo=$4
use_only_recent=$5
repo=$6
num=$7 # the number of iteration - 1 
use_only_previous=$8

hyopt=0 #1

data_dest="feature_data"
weighted=0
foldinfo_dest="data_info"
induc_dest="fix_induc"

mkdir -p logs/baseline

for bal_method in 'None'
# for bal_method in 'undersample'
do
	for seed in $(seq 0 $num)
	do
		echo "\n===============================================${seed}======================================================"	
		python3 train_dp_model.py -ccm_file ${data_dest}/shaped/${project}.ccms.ir_sim_diff.${weighted}.30.Days.json -dest results/base/${project}/${algo}/${bal_method}/ -project ${project} -model_type ${algo} -N $N -scale ${use_scale} -use_only_recent ${use_only_recent} -hyopt ${hyopt} -bal_method ${bal_method} -key base.${bal_method}.${algo}.N${N}.${seed} -seed ${seed} -foldinfo_dest ${foldinfo_dest} -repo ${repo} -is_base 1 -use_only_previous ${use_only_previous} -induc_dest ${induc_dest} > logs/baseline/${project}.${bal_method}.N${N}.${use_only_recent}.${seed}.out  &
		wait $!
	done
done
