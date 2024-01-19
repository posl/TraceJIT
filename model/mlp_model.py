"""
DP models based on MLP
"""
from sklearn.neural_network import MLPClassifier

class MLP_model(object):
	"""docstring for MLP_model"""
	def __init__(self, 
		hidden_layer_sizes = (100,),
		solver = 'adam',
		max_iter = 100,
		learning_rate_init = 0.001,
		learning_rate = 'constant',
		random_state = None):

		super(MLP_model, self).__init__()
		self.hidden_layer_sizes = hidden_layer_sizes
		self.solver = solver
		self.max_iter = max_iter
		self.learning_rate_init = learning_rate_init
		self.learning_rate = learning_rate # constant, invscaling, adaptive
		self.random_state = random_state

		# generate rgr
		self.mdl = self.generate(hidden_layer_sizes,
			solver, 
			max_iter, 
			learning_rate_init, 
			learning_rate, 
			random_state)


	def generate(self, 
		hidden_layer_sizes, 
		solver, 
		max_iter, 
		learning_rate_init, 
		learning_rate, 
		random_state,
		from_own = True, 
		opt = False):
		"""
		"""
		if from_own:
			clf = MLPClassifier(
				hidden_layer_sizes = hidden_layer_sizes,
				solver = self.solver,
				max_iter = self.max_iter,
				learning_rate_init = self.learning_rate_init,
				learning_rate = self.learning_rate,
				random_state = self.random_state,
				verbose = 1)
		else:
			clf = MLPClassifier(
				hidden_layer_sizes = hidden_layer_sizes,
				solver = solver,
				max_iter = max_iter,
				learning_rate_init = learning_rate_init,
				learning_rate = learning_rate,
				random_state = random_state,
				verbose = 1)

		return clf


	def fit_model(self, features, labels, opt = False, num_cv = 3, params = None):
		"""
		"""
		if opt:
			mdl = MLPClassifier()
			print ("num_cv", num_cv)
			self.mdl = self.hyperparam_optimise(features, 
				labels, 
				mdl,
				num_cv = num_cv)
#			self.mdl = MLPClassifier(
#				n_estimators = best_params['n_estimators'],
#				max_depth = None,#best_params['max_depth'],
#				min_samples_split = best_params['min_samples_split'],
#				random_state = None,
#				)
		else:
			if params is not None:
				self.mdl = self.generate(params['hidden_layer_sizes'],
					params['solver'], 
					params['max_iter'], 
					params['learning_rate_init'], 
					params['learning_rate'], 
					params['random_state'],
					from_own = False)
				
			self.mdl.fit(features, labels)
		return self.mdl


	def hyperparam_optimise(self, 
		train_feature_arr, 
		train_label_arr, model, num_cv = 3):
		"""
		"""
		pass
#		from sklearn.model_selection import GridSearchCV
#		import numpy as np	
#		from sklearn import metrics#

#		scoring_funcs = {'Brier':'brier_score_loss', 
#			'AUC':'roc_auc',
#			'F1':'f1'}
#	
#		param_grid = {
#			#'max_depth':list(np.int32(np.linspace(1,13,13))) + [None],
#			'max_depth':list(np.int32(np.linspace(1,15,15))) + [None], # for with fl
#			'min_samples_split':np.int32(np.linspace(2,10,5)),
#			'n_estimators':[100,150,200],
#			'random_state':[self.random_state]
#		}#

#		grid_search = GridSearchCV(estimator = model, 
#			param_grid = param_grid, 
#			cv = num_cv, verbose = 2,
#			refit = 'Brier',
#			scoring = scoring_funcs)#

#		indices = np.arange(len(train_label_arr))
#		np.random.shuffle(indices)#

#		print ("===", type(grid_search))
#		grid_search.fit(train_feature_arr[indices], train_label_arr[indices])	#

#		best_params = grid_search.best_params_
#		print ("BEST PARAMS", best_params)
#		#sys.exit()
#		return grid_search



	def predict(self, 
		features, 
		predtype = 'label'):
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

