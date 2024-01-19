#!/bin/bash

use_scale=$1
N=$2
project=$3
algo=$4
use_only_recent=$5
max_days=$6
seed=$7
repo=$8
use_only_previous=$9

hyopt=0 #1

is_weighted='not_weighted'
weighted=0

data_dest="feature_data" 
foldinfo_dest="data_info/"
induc_dest="fix_induc"

mkdir -p logs/w_irfl

for bal_method in 'None'
# for bal_method in 'undersample'
do
	echo "\n===============================================${seed}======================================================"

	python3 train_dp_model.py -ccm_file ${data_dest}/shaped/${project}.ccms.ir_sim_diff.${weighted}.${max_days}.Days.json -dest results/w_irfl/${project}/${algo}/${bal_method}/${max_days} -project ${project} -model_type ${algo} -N $N -scale ${use_scale} -use_only_recent ${use_only_recent} -hyopt ${hyopt} -bal_method ${bal_method} -key diff.${is_weighted}.${bal_method}.${algo}.N${N}.${seed} -seed ${seed} -foldinfo_dest ${foldinfo_dest} -repo ${repo} -use_only_previous ${use_only_previous} -induc_dest ${induc_dest} > logs/w_irfl/${project}.${bal_method}.${is_weighted}.N${N}.${use_only_recent}.${max_days}.${seed}.out &
	wait $!
done
