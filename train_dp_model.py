"""
Generate defect prediction model
Testing
"""
PROJECTS = ['Lang', 'Math']
SKIPPED_FILES = {'Lang':'feature_data/skipped/Lang.skipped',
    'Math':'feature_data/skipped/Math.skipped'}
WO_ANY_ADDED = True  # exclude those changes without any added lines

import numpy as np
import time
import json
import pandas as pd

def is_without_added(commit, repo, g = None, file_postfix = None):
    """
    return True if it is without any added line
    """
    if g is None:
        import git
        g = git.Git(repo)

    showinfo = g.show("--numstat", "--oneline", '-w', commit)
    show_lines = [line for line in showinfo.split("\n") if bool(line.strip())]

    for line in show_lines[1:]:
        vals = [v for v in line.split("\t") if bool(v)]
        assert len(vals) == 3, vals

        if vals[-1].endswith(file_postfix)\
            and vals[0].isdigit() and int(vals[0]) > 0:
            return False # contain addition

    return True

def get_target_data(target_commits, commits, feature_arr, label_arr):
    indices_to_target = [commits.index(t_c) for t_c in target_commits]
    ret_feature_arr = feature_arr[indices_to_target]
    ret_label_arr = label_arr[indices_to_target]
    return ret_feature_arr, ret_label_arr


def scale_ccm(ccm_feature_arr, scale_type = 'min-max'):
    """
    scaler ccm_feature_arr between 0 and 1
        -> use Min-Max scaler
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    if scale_type == 'min-max':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    scaled =  scaler.fit_transform(ccm_feature_arr)
    return scaled


def gen_indices_based_on_order(indices_to_target, 
    N = 5, use_only_recent = False, labels = None):
    """
    Each fold consists of indices_to_training_data and the corresponding indices_to_test_data
    """
    chunks = np.array_split(indices_to_target, N)
    start_indices = list(map(lambda chunk:chunk[0], chunks))	
    last_indices = list(map(lambda chunk:chunk[-1], chunks))
    
    n_folds = []
    for i in range(N - 1):
        if not use_only_recent:
            curr_first_fold = len(indices_to_target) - 1 - indices_to_target[start_indices[0]:last_indices[i]+1] 
            curr_scnd_fold = len(indices_to_target) - 1 - indices_to_target[start_indices[i+1]:last_indices[i+1]+1] 
        else:# only recent (only the closet fold)
            curr_first_fold = len(indices_to_target) - 1 - indices_to_target[start_indices[i]:last_indices[i]+1] 
            curr_scnd_fold = len(indices_to_target) - 1 - indices_to_target[start_indices[i+1]:last_indices[i+1]+1] 

        print ("Fold {}:".format(i))
        print ("index", curr_first_fold[0], curr_scnd_fold[0])
        print ("\tFor train: fix-induc: {}, non-induc: {}".format(
            np.sum(labels[curr_first_fold]), len(curr_first_fold)-np.sum(labels[curr_first_fold])))
        print ("\tFor test: fix-induc: {}, non-induc: {}".format(
            np.sum(labels[curr_scnd_fold]), len(curr_scnd_fold)-np.sum(labels[curr_scnd_fold])))
        
        n_folds.append([curr_first_fold, curr_scnd_fold]) # curr_first_fold -> for training, curr_scnd_fold -> for testing

    return n_folds


def evaluate(is_clf, predcs, labels, metric_type = 'auc'):
    """
    """
    from sklearn import metrics

    if metric_type == 'auc':
        if is_clf:
            #pos_predc_probas = list(map(lambda v:v[1], predcs))
            pos_predc_probas = np.asarray(predcs)[:,1]
        else:
            pos_predc_probas = predcs

        if len(set(labels)) == 1:
            return -1.
        else:
            return metrics.roc_auc_score(labels, pos_predc_probas)
    elif metric_type == 'precision':
        if 1 in predcs: # should predict at least one postive class
            #if is_clf:
            return metrics.precision_score(labels, predcs)
        else:
            return -1.
        #else:
        #	return 0.0
    elif metric_type == 'precision_formula':
        true_positives = sum(1 for t, p in zip(labels, predcs) if t == 1 and p == 1)
        false_positives = sum(1 for t, p in zip(labels, predcs) if t == 0 and p == 1)
        
        return f'{true_positives}/{(true_positives + false_positives)}'
    
    elif metric_type == 'recall':
        #if is_clf:
        if 1 in labels: # should hava at least one positive class (1)
            return metrics.recall_score(labels, predcs)
        else:
            return -1.
        #else:
        #	return 0.0
    elif metric_type == 'f1':
        #if is_clf:
        try:
            return metrics.f1_score(labels, predcs)
        except Exception as e:
            print (e)
            return -1.
        #else:
        #	return 0.0
    elif metric_type == 'acc':
        return metrics.accuracy_score(labels, predcs)
    elif metric_type == 'bal_acc':
        return metrics.balanced_accuracy_score(labels, predcs)
    elif metric_type == 'tn_rate':
        tn, fp, fn, tp = metrics.confusion_matrix(labels, predcs).ravel()
        tn_rate = tn/(tn + fp)
        return tn_rate
    elif metric_type == 'pr_auc':
        if is_clf:
            #pos_predc_probas = list(map(lambda v:v[1], predcs))
            pos_predc_probas = np.asarray(predcs)[:,1]
        else:
            pos_predc_probas = predcs
        
        pr_auc_score = metrics.average_precision_score(labels, pos_predc_probas)

        if np.isnan(pr_auc_score):
            pr_auc_score = -1.

        return pr_auc_score
    elif metric_type == 'brier':
        if is_clf:
            #pos_predc_probas = list(map(lambda v:v[1], predcs))
            pos_predc_probas = np.asarray(predcs)[:,1]
        else:
            pos_predc_probas = predcs
        return metrics.brier_score_loss(labels, pos_predc_probas) #smaller -> better
    else:
        print ("Currenlty %s is not supported" % metric_type)
        import sys
        sys.exit()


def predict_and_evaluate(is_clf,
    model, 
    feature_arr, 
    label_arr, 
    stat_file_path = None, 
    save = True):
    """
    evaluation
    """
    _predictions = model.predict(feature_arr, predtype = 'label')	
    _predc_probas = model.predict(feature_arr, predtype = 'prob')

    if is_clf: # logging
        print ("Only for CLF")
        flag_to_pos = label_arr == 1 
        if not (any(flag_to_pos)):
            print ("\tNo fix-inducing")

        flag_to_neg = label_arr == 0 
        if not (any(flag_to_neg)):
            print ("\tNo non-fix-inducing")

        print ("\tTotal: {} (fix-inducing) vs {} (non-inducing)".format(np.sum(flag_to_pos), np.sum(flag_to_neg)))

    auc_roc = evaluate(is_clf, _predc_probas, label_arr, metric_type = 'auc')
    prec = evaluate(is_clf, _predictions, label_arr, metric_type = 'precision')
    prec_formula = evaluate(is_clf, _predictions, label_arr, metric_type = 'precision_formula')
    recall = evaluate(is_clf, _predictions, label_arr, metric_type = 'recall')
    f1 = evaluate(is_clf, _predictions, label_arr, metric_type = 'f1')

    brier_score = evaluate(is_clf, _predc_probas, label_arr, metric_type = 'brier')

    acc = evaluate(is_clf, _predictions, label_arr, metric_type = 'acc')
    balanced_acc = evaluate(is_clf, _predictions, label_arr, metric_type = 'bal_acc')
    tn_rate = evaluate(is_clf, _predictions, label_arr, metric_type = 'tn_rate')
    pr_auc_score = evaluate(is_clf, _predc_probas, label_arr, metric_type = 'pr_auc')

    print ("\tPer class: 1(%d), 0(%d)" % (list(label_arr).count(1), list(label_arr).count(0)))
    print ("\tAUC: %f" % auc_roc)
    print ("\tPrecision: %f" % prec)
    print ("\tRecall: %f" % recall)
    print ("\tF1: %f" % f1)
    print ("\tBrier Score: %f" % brier_score)
    print ("\tAcc: %f" % acc)	
    print ("\tBal_Acc: %f" % balanced_acc)	
    print ("\tTN_rate: %f" % tn_rate)	
    print ("\tPR_score: %f" % pr_auc_score)	
    print(f'\tPrecision_formula: {prec_formula}')
    
    if stat_file_path is not None and save:
        stat_output_df = pd.DataFrame({'AUC':[auc_roc], 'Precision':[prec], 'Recall':[recall], 'F1':[f1], 
        'Brier':[brier_score], 'Acc':[acc], 'Bal_Acc':[balanced_acc], 'TN_Rate':[tn_rate], 'PR_score':[pr_auc_score]})
        stat_output_df.to_csv(stat_file_path)
        
    return _predictions, _predc_probas, label_arr


def balanced(feature_arr, label_arr, method = 'undersample'):
    """
    apply smote
    """
    if method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        print ("Before {}, counts of label '0': {}".format(method, sum(label_arr == 0.)))
        print ("Before {}, counts of label '1': {}".format(method, sum(label_arr == 1.)))
        rus = RandomUnderSampler(random_state=0, sampling_strategy = 0.2)
        feature_arr_res, label_arr_res = rus.fit_resample(feature_arr, label_arr)

        print ("After {}, counts of label '0': {}".format(method, sum(label_arr_res == 0.)))
        print ("After {}, counts of label '1': {}".format(method, sum(label_arr_res == 1.)))
        return feature_arr_res, label_arr_res

    else: #
        from imblearn.over_sampling import SMOTE
        print ("Before {}, counts of label '0': {}".format(method, sum(label_arr == 0.)))
        print ("Before {}, counts of label '1': {}".format(method, sum(label_arr == 1.)))

        sm = SMOTE(random_state = 2, sampling_strategy = 1.0)
        feature_arr_res, label_arr_res = sm.fit_sample(feature_arr, label_arr)

        print ("After {}, counts of label '0': {}".format(method, sum(label_arr_res == 0.)))
        print ("After {}, counts of label '1': {}".format(method, sum(label_arr_res == 1.)))

        return feature_arr_res, label_arr_res


#def get_feature_data(ccm_file, label_file, is_base = False, with_scale = False):
def get_feature_data(ccm_file, is_base = False, with_scale = False, with_tjit = False):
    """
    """
    with open(ccm_file) as f:
        data = json.load(f)
    #    raw_data = json.load(f)
    #with open(label_file) as f:
    #    labels = json.load(f)

	#data = utils.convert_to_arr(raw_data, labels)
    ccm_feature_arr = np.asarray(data['features'])
    labeled_commits = data['commits_and_labels']
    label_arr = np.asarray([vs[1] for vs in labeled_commits], dtype = int)
    commits = [vs[0] for vs in labeled_commits]
    headers = data['headers']
    
    if is_base:
        ccm_feature_arr = ccm_feature_arr[:,:-3]
        headers = headers[:-3]

    if with_tjit:
        #ccm_feature_arr = ccm_feature_arr[:, :-6]
        #headers = headers[:-6]
        # m_list = [-7, -6, -5] # TL_UL, ATL_UL, DTL_UL
        m_list = [-7, -6, -5, -4, -3, -2, -1] #全て使う
        # m_list = [-3, -4, -7, -8, -9, -10]
        # m_list = [-3, -4, -7]
        #ccm_feature_arr = np.concatenate([ccm_feature_arr[:, :13], ccm_feature_arr[:, m_no].reshape(-1, 1)], axis=1)
        #headers = headers[:13] + [headers[m_no]]
        
        ccm_feature_arr = np.concatenate([ccm_feature_arr[:, :13], ccm_feature_arr[:, m_list]], axis=1)
        headers = headers[:13] + [headers[no] for no in m_list]
        
            
    if not is_base and not with_tjit:
        ccm_feature_arr = ccm_feature_arr
        headers = headers
    
    if with_scale:
        ccm_feature_arr = scale_ccm(ccm_feature_arr, scale_type = 'min-max')
    
    return headers, commits, label_arr, ccm_feature_arr 	

def is_include_NaN(ccm_feature_arr):
    """
    check NaN
    """
    import math
    for feature_row in ccm_feature_arr:
        for v in feature_row:
            if math.isnan(v):
                print ("NaN is not a valid value")
                return True
    return False


def exclude_wo_any_addition(repo, skipped_file, target_commits):
    """
    exclude commits without any added line (and recored those commits in "skipped_file")
    """
    if not os.path.exists(skipped_file):
        import git 
        g = git.Git(repo)
        filtered = []
        skipped = []
        for c in target_commits:
            is_commit_without_addedd_flag = is_without_added(c, None, g = g, file_postfix = ".java")
            if not bool(is_commit_without_addedd_flag):
                filtered.append(c)	
            else:
                skipped.append(c)	

        target_commits = filtered
        # save skipped commits 
        with open(skipped_file, 'w') as f:
            for c in skipped:
                f.write(c + "\n")
    else:
        if os.stat(skipped_file).st_size > 0:
            skipped = pd.read_csv(skipped_file, header = None)[0].values
            target_commits = [c for c in target_commits if c not in skipped]
            print ("{} -> {}".format(len(commits), len(target_commits)))
        
    return target_commits

def change_labels(fix_and_inducing_file, commits, label_arr):
    with open(fix_and_inducing_file) as f:
        induc_data = json.load(f)
    induc_commit_list = [set() for i in range(len(label_arr))]
    for i, label in enumerate(label_arr):
        if bool(label):
            commit = commits[i]
            for id in induc_data:
                if id[1] == commit:
                    induc_commit_list[i].add(id[0])
    return induc_commit_list

if __name__ == "__main__":
    import argparse
    import os, sys
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("-ccm_file", type = str,
        help = "a path to a file that stores feature data with labels and commits")
    parser.add_argument("-dest", type = str,
        help = "a path to the destination directory")
    parser.add_argument("-project", type = str, 
        help = "a target project ID")
    parser.add_argument("-model_type", type = str, 
        help = "RF, MLP, SVM", default = 'RF')
    parser.add_argument("-N", type = int,
        help = "N folds cross validataion")
    parser.add_argument("-repo", type = str,
        help = "a path to the git repository of the target project")
    parser.add_argument("-seed", type = int, default = None,
        help = "seed for random")
    parser.add_argument("-key", type = str, default = "0",
        help = "a key to the current run")
    parser.add_argument("-scale", type = int, default = 1,
        help = "1 for scale 0 for using raw")
    parser.add_argument("-use_only_recent", type = int, default = 0,
        help = "1 if using only the most recent fold to test the closet following fold")
    parser.add_argument("-hyopt", type = int, default = 0)
    parser.add_argument("-bal_method", type = str, default = None)
    parser.add_argument("-foldinfo_dest", type = str, default = ".")
    parser.add_argument("-is_base", type = int, default = 0, 
        help = "1 if this is for generating baselines, otherwise 0")
    parser.add_argument("-use_only_previous", type = int, default = 0,
        help = "1 if using only the previous reported data")
    parser.add_argument("-induc_dest", type=str, default='.')
    parser.add_argument("-with_tjit", type = int, default = 0,
        help = "i if using tjit metrics")
    #parser.add_argument("-gt_file", type = str, 
	#	help = "a file that contains the ground-truth of fix-inducing commits. \
	#		Here, we expect the file to contain a dictionary with commits as key and 1(fix-inducing) or 0(non-inducing) as value")

    
    args = parser.parse_args()

    assert args.project in PROJECTS, "{} is not among our targets".format(args.project)

    dest = args.dest
    os.makedirs(dest, exist_ok = True)
    os.makedirs(os.path.join(dest, "stat"), exist_ok = True)
    os.makedirs(os.path.join(dest, "output"), exist_ok = True)

    key = args.key
    #headers, commits, label_arr, ccm_feature_arr = get_feature_data(args.ccm_file, args.gt_file, is_base = bool(args.is_base), with_scale = False)
    headers, commits, label_arr, ccm_feature_arr = get_feature_data(args.ccm_file, is_base = bool(args.is_base), with_scale = False, with_tjit=bool(args.with_tjit))


    assert not is_include_NaN(ccm_feature_arr), 'NaN in ccm feature data'

    t1 = time.time()
    target_commits = commits # use all
    

    ##### filter out without any added line
    if WO_ANY_ADDED:
        target_commits = exclude_wo_any_addition(args.repo, SKIPPED_FILES[args.project], target_commits)

    print ("Key is {}".format(key))
    t2 = time.time()
    print ("Time taken for excluding out of our target range commit: %f" % (t2 - t1))

    target_commit_feature_arr, target_label_arr = get_target_data(target_commits, commits, ccm_feature_arr, label_arr)

    foldinfo_dest = args.foldinfo_dest
    use_only_previous = args.use_only_previous
    if bool(use_only_previous):
        fix_and_inducing_file = os.path.join(args.induc_dest, args.project)
        fix_and_inducing_file = os.path.join(fix_and_inducing_file, "fix_and_introducers_pairs.json")
        inducing_commits_list = change_labels(fix_and_inducing_file, target_commits, target_label_arr)
        print(inducing_commits_list)
        target_changed_label_arr = np.full((5, len(target_commits)), 0)

    # retrieve (and save) N-fold cross validation information
    if args.use_only_recent:
        foldinfo_dest = os.path.join(foldinfo_dest, 'only_recent')
    os.makedirs(foldinfo_dest, exist_ok = True)

    train_and_test_file = os.path.join(foldinfo_dest, 
        "{}.train_and_test.{}.Nfold.json".format(args.project, args.N))

    if os.path.exists(train_and_test_file):
        with open(train_and_test_file) as f:
            train_and_test_lst = json.load(f)

        n_folds = []
        for fold_key, train_and_test_commits in train_and_test_lst.items():
            fold_key = int(fold_key)
            fold = str(fold_key)
            pairs = []
            for c in train_and_test_commits['train']:
                pairs.append(target_commits.index(c))
                #ラベルの付け替え
                if bool(use_only_previous):
                    if target_label_arr[target_commits.index(c)] == 1:
                        for induc in inducing_commits_list[target_commits.index(c)]:
                            if induc in train_and_test_lst[fold]['train']:
                                target_changed_label_arr[fold_key][target_commits.index(c)] = 1
                                break
                            else:
                                target_changed_label_arr[fold_key][target_commits.index(c)] = 0
            pairs = [[target_commits.index(c) for c in train_and_test_commits['train']]]
            pairs.append([target_commits.index(c) for c in train_and_test_commits['test']])#
            n_folds.append(pairs)
    else:
        # split to N folds
        indices_to_target = np.arange(0, len(target_commits), 1) 
        os.makedirs(foldinfo_dest, exist_ok = True)	

        # split to N folds
        n_folds = gen_indices_based_on_order(indices_to_target,
            N = args.N,
            use_only_recent = bool(args.use_only_recent),
            labels = target_label_arr)

        with open(train_and_test_file, 'w') as f:
            as_dict = {}
            for i, n_fold in enumerate(n_folds):
                as_dict[i] = {}
                as_dict[i]['train'] = [target_commits[idx_to_commit] for idx_to_commit in n_fold[0]]
                as_dict[i]['test'] = [target_commits[idx_to_commit] for idx_to_commit in n_fold[1]]	
            f.write(json.dumps(as_dict))
            #ラベルの付け替え
            if bool(use_only_previous):
                for fold_key, train_and_test_commits in as_dict.items():
                    fold_key = int(fold_key)
                    fold = str(fold_key)
                    for c in train_and_test_commits['train']:
                        if target_label_arr[target_commits.index(c)] == 1:
                            for induc in inducing_commits_list[target_commits.index(c)]:
                                if induc in train_and_test_commits[fold]['train']:
                                    target_changed_label_arr[fold_key][target_commits.index(c)] = 1
                                    break
                                else:
                                    target_changed_label_arr[fold_key][target_commits.index(c)] = 0

        print ("Write to {}".format(train_and_test_file))

    
    ## now, train and evaluate a DP model
    # generate empty model
    if args.model_type == "RF": 
        ###########################################################
        #### a leanring algorithm used in SANER'21 experiments ####
        ###########################################################
        from model.rf_model import RF_model
        print ("Here Random Forest Classifier")
        model = RF_model(
            n_estimators = 200,
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            random_state = args.seed)
        is_clf = True
    elif args.model_type == 'RFR':
        from model.rf_rgr_model import RFR_model
        print ("Here Random Forest Regressor")
        model = RFR_model(
            n_estimators = 100,
            max_depth = None,
            min_samples_split = 2,
            random_state = args.seed)
        is_clf = False
    elif args.model_type == 'MLP':
        from model.mlp_model_keras import MLP_model
        print ("Here MLP")
        model = MLP_model(
            target_commit_feature_arr.shape[1],
            hidden_layer_sizes = [32, 64],
            dropouts = [0.5, 0.3],
            batch_size = 256,
            epochs = 100,
            learning_rate = 1e-3,
            random_state = None)
        is_clf = False
    elif args.model_type == 'SVR':
        from model.svr_model import SVR_model
        print ("Here SVR")
        model = SVR_model(kernel = 'rbf', 
            gamma = 'scale', 
            epsilon = 0.1, 
            degree = 3, 
            C = 1.0,
            coef0 = 1, 
            verbose = True)
        is_clf = False
    elif args.model_type == 'SVC':
        from model.svc_model import SVC_model
        print ("Here SVC")
        model = SVC_model(kernel = 'rbf', 
            gamma = 'scale', 
            #epsilon = 0.1, 
            degree = 3, 
            C = 1.0,
            coef0 = 1, 
            verbose = True)
        is_clf = True
    elif args.model_type == 'LR':
        from model.LR_model import LogisticRegression_model
        print ("Here LR")
        model = LogisticRegression_model(random_state = args.seed,
            penalty = 'l2',
            solver = 'liblinear',#newton-cg',
            verbose = True,
            max_iter = 200,
            C = 1.0)
        is_clf = True
    elif args.model_type == 'PolyLR':
        from model.poly_linear_rg_model import PolynomialRegression_model
        print ("Here PR")
        model = PolynomialRegression_model(fit_intercept = True,
            degree = 2)
        is_clf = True
    elif args.model_type == 'LDA':
        from model.LDA_model import LDA_model
        print ("Here LDA")
        n_components = 1 # since there is only fix-inducing and non-fix-inducing
        model = LDA_model(n_components, 
            solver = 'eigen',
            tol = 0.0001)
        is_clf = True
    elif args.model_type == 'GP':
        from model.gp_model_in_test import GP_model
        print ("Here GP")
        model = GP_model(
            headers,
            maxTreeDepth = 8, 
            minTreeDepth = 1, 
            initMaxTreeDepth = 6, 
            cxpb = 0.8,
            mutpb = 0.2, 
            random_state = 0,
            num_pop = 50,
            ngen = 200,
            is_pareto = True,
            use_dist_as_fit = False,
            use_min_max_scaler = False)
        is_clf = False
    else:
        print ("Not supported: {}".format(args.model_type))
        sys.exit()

    # main training part
    predictions = []
    predc_probas = []
    ys = []
    outputs = {'commit':[], 'pred':[], 'prob':[], 'label':[], 'fold':[]}

    for i, (indices_to_train, indices_to_test) in enumerate(n_folds):
        # fit the model on the training data
        print ('\n=======================Current fold: %d==============================' % i)
        if bool(use_only_previous):
            print ('Train: 1:{}, 0:{}'.format(list(target_changed_label_arr[i][indices_to_train]).count(1), 
                list(target_changed_label_arr[i][indices_to_train]).count(0)))
        else:
            print ('Train: 1:{}, 0:{}'.format(list(target_label_arr[indices_to_train]).count(1), 
                list(target_label_arr[indices_to_train]).count(0)))
        print ('Test: 1:{}, 0:{}'.format(list(target_label_arr[indices_to_test]).count(1), 
            list(target_label_arr[indices_to_test]).count(0)))
        if bool(use_only_previous):
            train_label_arr = target_changed_label_arr[i][indices_to_train]
        else:
            train_label_arr = target_label_arr[indices_to_train]
        train_feature_arr = target_commit_feature_arr[indices_to_train]
        if bool(args.scale): 
            train_feature_arr = scale_ccm(train_feature_arr, scale_type = 'min-max')

        if args.bal_method is not None and args.bal_method != 'None':
            new_train_feature_arr, new_train_label_arr = balanced(
                train_feature_arr, 
                train_label_arr, 
                method = args.bal_method)
        else:
            new_train_feature_arr = train_feature_arr
            new_train_label_arr = train_label_arr

        #num_cv = int(np.ceil(len(new_train_feature_arr) / 500))
        #num_cv = 2 if num_cv < 2 else num_cv
        num_cv = None
        if args.model_type == 'MLP':
            print (new_train_feature_arr.shape)
            model.fit_model(new_train_feature_arr, new_train_label_arr)
        elif args.model_type == 'GP':
            model.fit_model(
                new_train_feature_arr,
                new_train_label_arr, 
                num_of_best = 1)
        else:
            model.fit_model(new_train_feature_arr,
                new_train_label_arr,
                opt = bool(args.hyopt),
                num_cv = num_cv)	

        if True:
            # save current model
            mdl_dest = os.path.join(dest, "model")
            os.makedirs(mdl_dest, exist_ok=True)
            save_path = os.path.join(mdl_dest, "%s.%s.%d.%s.sav" % (args.project, args.model_type, i, key))
            print ("Key is {}".format(key))
            print ("save to {}".format(save_path))

            if args.model_type == 'MLP':
                model.mdl.save(save_path) 
            elif args.model_type == 'GP':
                print (model.mdl)
                with open(save_path, 'w') as f:
                    f.write(str(model.mdl) + "\n")
            else:
                with open(save_path, 'wb') as f:
                    pickle.dump(model.mdl, f)

        print ("\n=================Local (per-fold) Statistics for training data: fold %d =====================" % i)
        predict_and_evaluate(
            is_clf,
            model, 
            train_feature_arr, 
            train_label_arr)

        print ("\n=================Local (per-fold) Statistics for test data: fold %d =====================" % i)
        test_feature_arr = target_commit_feature_arr[indices_to_test]
        if bool(args.scale):
            test_feature_arr = scale_ccm(test_feature_arr, scale_type = 'min-max')
        
        stat_file_path = os.path.join(dest, "stat/%s.%s.%d.%s.stat" % (args.project, args.model_type, i, key))
        print(f'test_feature: {test_feature_arr[:10]}')
        print(f'test_label: {target_label_arr[indices_to_test][:10]}')
        _predictions, _predc_probas, _ys = predict_and_evaluate(
            is_clf,
            model, 
            test_feature_arr, 
            target_label_arr[indices_to_test], 
            stat_file_path = stat_file_path,
            save = not bool(args.hyopt))

        predictions.extend(list(_predictions))
        predc_probas.extend(list(_predc_probas))

        ys.extend(list(_ys))
        outputs['commit'].extend(train_and_test_lst[str(i)]['test'])
        outputs['pred'].extend(_predictions)
        if is_clf:
            outputs['prob'].extend(list(map(lambda v:v[1], _predc_probas)))
        else:
            outputs['prob'].extend(_predc_probas)
        outputs['label'].extend(_ys)
        outputs['fold'].extend([i] * len(_ys))
        
        ## save output per fold
        output_file_path = os.path.join(dest, "output/%s.%s.%d.%s.csv" % (args.project, args.model_type, i, key))
        output = {}
        output['commit'] = train_and_test_lst[str(i)]['test']
        output['pred'] = _predictions
        if is_clf:
            output['prob'] = list(map(lambda v:v[1], _predc_probas))
        else:
            output['prob'] = _predc_probas
        output['label'] = _ys
        output['fold'] = [i] * len(_ys)
        df_output = pd.DataFrame.from_dict(output)
        df_output.to_csv(output_file_path) 

    # summary over the entire test dataset
    print ('\nModel classes:', set(target_label_arr)) 
    if True:
        auc_roc = evaluate(is_clf, predc_probas, ys, metric_type = 'auc')	
        prec = evaluate(is_clf, predictions, ys, metric_type = 'precision')
        recall = evaluate(is_clf, predictions, ys, metric_type = 'recall')	
        f1 = evaluate(is_clf, predictions, ys, metric_type = 'f1')	

        brier_score = evaluate(is_clf, predc_probas, ys, metric_type = 'brier')

        acc = evaluate(is_clf, predictions, ys, metric_type = 'acc')
        balanced_acc = evaluate(is_clf, predictions, ys, metric_type = 'bal_acc')
        tn_rate = evaluate(is_clf, predictions, ys, metric_type = 'tn_rate')
        pr_auc_score = evaluate(is_clf, predc_probas, ys, metric_type = 'pr_auc')
        prec_formula = evaluate(is_clf, predictions, ys, metric_type = 'precision_formula')

        print ("\n=================Statistics (over the entire test dataset)=====================")	

        print ("\tAUC: %f" % auc_roc)
        print ("\tPrecision: %f" % prec)
        print ("\tRecall: %f" % recall)
        print ("\tF1: %f" % f1)	
        print ("\tBrier Score: %f" % brier_score)
        print ("\tAcc: %f" % acc)	
        print ("\tBal_Acc: %f" % balanced_acc)	
        print ("\tTN_rate: %f" % tn_rate)	
        print ("\tPR_score: %f" % pr_auc_score)	
        print (f'\tPrecision_formula: {prec_formula}')

        stat_file_path = os.path.join(dest, "stat/%s.%s.all.%s.stat" % (args.project, args.model_type, key))
        with open(stat_file_path, 'w') as f:
            f.write("AUC,Precision,Recall,F1,Brier,Acc,Bal_Acc,TN_Rate,PR_score, Precision_fomula\n")
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                auc_roc, prec, recall, f1, brier_score, acc, balanced_acc, tn_rate, pr_auc_score, prec_formula))

    # save prediction results
    output_file_path = os.path.join(dest, "output/%s.%s.%s.csv" % (args.project, args.model_type, key))
    df_outputs = pd.DataFrame.from_dict(outputs)
    df_outputs.to_csv(output_file_path) 



