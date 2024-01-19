"""
DP model based on RandomForest
"""
GRIDPARAMS = {'RF':['max_depth', 'min_samples_split', 'n_estimators']}
from sklearn.ensemble import RandomForestClassifier

class RF_model(object):
	"""docstring for RF_model"""
	def __init__(self, n_estimators = 200, #100 
		max_depth = None, 
		min_samples_split = 3,
		min_samples_leaf = 2,
		max_features = 'sqrt',
		random_state = None):

		super(RF_model, self).__init__()
	
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.random_state = random_state
		self.max_features = max_features

		# generate clf
		self.mdl = self.generate(n_estimators, 
			max_depth, 
			min_samples_split, 
			min_samples_leaf, 
			random_state)


	def generate(self, 
		n_estimators, 
		max_depth, 
		min_samples_split, 
		min_samples_leaf, 
		random_state, 
		from_own = True, 
		opt = False):
		"""
		"""
		if from_own:
			clf = RandomForestClassifier(n_estimators = self.n_estimators, 
				max_depth = self.max_depth,
				min_samples_split = self.min_samples_split,
				min_samples_leaf = self.min_samples_leaf,
				random_state = self.random_state,
				max_features = self.max_features,
				#class_weight = 'balanced',
				criterion = "gini", #"entropy",
				verbose = 1)

			#from imblearn.ensemble import BalancedRandomForestClassifier
			#clf = BalancedRandomForestClassifier(n_estimators = self.n_estimators, 
			#	max_depth = self.max_depth,
			#	min_samples_split = self.min_samples_split,
			#	min_samples_leaf = self.min_samples_leaf,
			#	random_state = self.random_state,
			#	max_features = self.max_features,
			#	#class_weight = 'balanced',
			#	criterion = "gini", #"entropy",
			#	verbose = 1)
		else:
			clf = RandomForestClassifier(n_estimators = n_estimators, 
				max_depth = max_depth,
				min_samples_split = min_samples_split,
				min_samples_leaf = min_samples_leaf,
				random_state = random_state,
				max_features = self.max_features,
				#class_weight = 'balanced',
				criterion = "gini", #"entropy",
				verbose = 1)		

		return clf


	def fit_model(self, features, labels, opt = False, num_cv = 3, params = None):
		"""
		"""
		if opt:
			mdl = RandomForestClassifier()
			print ("num_cv", num_cv)
			self.mdl = self.hyperparam_optimise(
				features, 
				labels, 
				mdl,
				num_cv = num_cv)
		else:
			if params is not None:
				self.mdl = self.generate(params['n_estimators'], 
					params['max_depth'], 
					params['min_samples_split'], 
					params['min_samples_leaf'],
					self.random_state,
					from_own = False)
				
			self.mdl.fit(features, labels)
		return self.mdl


	def hyperparam_optimise(self, train_feature_arr, train_label_arr, model, num_cv = 3):
		"""
		"""
		from sklearn.model_selection import GridSearchCV
		import numpy as np	
		from sklearn import metrics

		#scoring_funcs = {'Brier':'brier_score_loss', 
		#	'AUC':'roc_auc',
		#	'F1':'f1'}
		scoring_funcs = {'AUC':'roc_auc', 'F1':'f1'}
	
		param_grid = {
			#'max_depth':list(np.int32(np.linspace(1,13,13))) + [None],
			#'max_depth':list(np.int32(np.linspace(1,15,15))) + [None], # for with fl
			#'max_depth':[1,3,5,7,9,11,13,15,'None'],
			#'max_depth':[10,15,'None'],
			'min_samples_split':[2, 5, 10], #np.int32(np.linspace(2,10,5)),
			'n_estimators':[int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
			'random_state':[self.random_state], 
			'max_features':['sqrt']#,
			#'bootstrap':[True, False]
		}

		grid_search = GridSearchCV(estimator = model, 
			param_grid = param_grid, 
			cv = num_cv, verbose = 2,
			refit = 'AUC',
			scoring = scoring_funcs)

		indices = np.arange(len(train_label_arr))
		np.random.shuffle(indices)

		print ("===", type(grid_search))
		grid_search.fit(train_feature_arr[indices], train_label_arr[indices])

		best_params = grid_search.best_params_
		print ("BEST PARAMS", best_params)
		#sys.exit()
		return grid_search



	def predict(self, features, predtype = 'label'):
		"""
		predtype: label, prob, log_prob
		"""

		if predtype == 'label':
			predictions = self.mdl.predict(features)
		elif predtype == 'prob':
			#print ("HERE")
			predictions = self.mdl.predict_proba(features)
		else:
			predictions = self.mdl.predct_log_prob(features)

		return predictions

