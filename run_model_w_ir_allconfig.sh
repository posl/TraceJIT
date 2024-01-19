#!/bin/bash

use_scale=$1
N=$2
project=$3
algo=$4
use_only_recent=$5
max_days=$6
repo=$7
num=$8
use_only_previous=$9

start_num=0
for seed in $(seq ${start_num} $num)
do
	./run_model_w_ir_final.sh ${use_scale} $N ${project} ${algo} ${use_only_recent} ${max_days} ${seed} ${repo} ${use_only_previous}
	wait $!
done