"""
"""
import argparse
import json
import os, sys
import glob
import numpy as np

INDICES = {'AUC':0,'Precision':1, 'Recall':2, 'F1':3, 'Brier':4}

def get_best_one(results, crt):
	""
	""
	idx_to_crt = INDICES[crt]
	if crt == 'Brier':
		reverse = True
	else:
		reverse = False

	vals_w_indices = []
	for i,val in results.items():
		vals_w_indices.append((i, val[idx_to_crt]))


	sorted_vals_w_indices = sorted(vals_w_indices, key = lambda v:v[1], reverse = reverse)
	print (sorted_vals_w_indices)
	idx_best_one, best_val = sorted_vals_w_indices[0]
	print ("In terms of {}, {} is the best: {}".format(crt, idx_best_one, best_val))

	return idx_best_one


parser = argparse.ArgumentParser()
parser.add_argument("-resultdir", type = str)
parser.add_argument("-project", type = str)
parser.add_argument("-bal_method", type = str, default = None, 
	help = "None, oversample")
parser.add_argument("-use_all", type = int, default = 0)
parser.add_argument("-algo", type = str, default = 'RF', 
	help = 'RF, LR')
parser.add_argument("-N", type = int, default = 5)

args = parser.parse_args()

print (args)
print (args.project)
INDICES_TO_BEST = {}
for i in range(args.N - 1):
	INDICES_TO_BEST[i] = {}

	targetfiles = glob.glob(os.path.join(args.resultdir, "{}.{}.{}.*.stat".format(
		args.project, args.algo, i)))

	results = {}
	for targetfile in targetfiles:
		with open(targetfile) as f:
			lines = [line.strip() for line in f.readlines()]

		target_results = lines[1]
		vals = [np.float16(v) for v in target_results.split("AUC")[0].split(",")]

		iter_num = int(os.path.basename(targetfile).split(".")[-2])
		
		results[iter_num] = vals

	iter_num_in_order = sorted(list(results.keys()))

	
	for crt in INDICES.keys():
		print (results)
		idx_best_one = get_best_one(results, crt)
		INDICES_TO_BEST[i][crt] = idx_best_one


print ("\n============================================================")
for idx_to_fold, best_ones in INDICES_TO_BEST.items():
	print ('For {} fold'.format(idx_to_fold))
	for crt, idx_to_best in best_ones.items():
		print ("\tFor {}, {}".format(crt, idx_to_best))