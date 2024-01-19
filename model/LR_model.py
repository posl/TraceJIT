"""
Logistic Regression model
"""
from sklearn.linear_model import LogisticRegression

class LogisticRegression_model(object):
	"""docstring for LogisticRegression_model"""

	def __init__(self, random_state = None,
		penalty = 'l2',
		solver = 'lbfgs',
		verbose = 1,
		max_iter = 100,
		C = 1.0):

		super(LogisticRegression_model, self).__init__()

		self.random_state = random_state
		self.penalty = penalty
		self.solver = solver
		self.verbose = verbose
		self.max_iter = max_iter
		self.C = C

		self.mdl = self.generate(random_state, 
			penalty, 
			solver, 
			verbose, 
			max_iter,
			C)

	def generate(self, 
		random_state, 
		penalty, 
		solver, 
		verbose, 
		max_iter,
		C, 
		from_own = True):
		"""
		"""

		if from_own:
			clf = LogisticRegression(random_state = self.random_state,
				penalty = self.penalty,
				solver = self.solver,
				verbose = self.verbose,
				max_iter = self.max_iter,
				C = self.C)
		else:
			clf = LogisticRegression(random_state = random_state,
				penalty = penalty,
				solver = solver,
				verbose = verbose,
				max_iter = max_iter,
				C = C)

		return clf


	def fit_model(self, features, labels, opt = True, num_cv = 3, params = None):
		"""
		"""
		if opt:
			model = LogisticRegression()
			print ("num_cv", num_cv)
			self.mdl = self.hyperparam_optimise(features, 
				labels, 
				model, 
				num_cv = num_cv)
#			self.mdl = LogisticRegression(random_state = self.random_state,
#				penalty = best_params['penalty'],
#				solver = best_params['solver'],
#				C = best_params['C'],
#				verbose = self.verbose,
#				max_iter = best_params['max_iter'])
		else:
			if params is not None:
				self.mdl = self.generate(self.random_state, 
					params['penalty'], 
					params['solver'], 
					True, 
					params['max_iter'],
					params['C'],
					from_own = False)

			self.mdl.fit(features, labels)
		return self.mdl


	def hyperparam_optimise(self, train_feature_arr, train_label_arr, model, num_cv = 3):
		"""
		"""
		from sklearn.model_selection import GridSearchCV
		import numpy as np	
		from sklearn import metrics

		scoring_funcs = {'Brier':'brier_score_loss', 
			'AUC':'roc_auc',
			'F1':'f1'}
	
		param_grid = {
			#'penalty':['l1'],
			'penalty':['l2'],
			'C':[0.1] + list(np.arange(1,10,2)) + [10],
			'solver':['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'], #'l2'
			#'solver':['liblinear', 'saga'], # 'l1'
			'max_iter':list(np.arange(500,1000,100)) + [1000],
			'random_state':[self.random_state]
		}

		grid_search = GridSearchCV(estimator = model, 
			param_grid = param_grid, 
			cv = num_cv, verbose = 2,
			refit = 'Brier',
			scoring = scoring_funcs)

		indices = np.arange(len(train_label_arr))
		np.random.shuffle(indices)

		print ("===", type(grid_search))
		grid_search.fit(train_feature_arr[indices], train_label_arr[indices])	

		#best_params = grid_search.best_params_
		print ("BEST PARAMS", grid_search.best_params_)
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
