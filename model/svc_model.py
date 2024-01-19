"""
Support Vector Regression (SVC) model
"""
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

class SVC_model(object):
	"""docstring for SVC_model"""
	def __init__(self, kernel = 'rbf', 
		gamma = 'scale', 
		#epsilon = 0.1, 
		degree = 3, 
		C = 1.0,
		coef0 = 1, 
		verbose = True):

		super(SVC_model, self).__init__()

		self.kernel = kernel
		self.gamma = gamma
		#self.epsilon = epsilon
		self.degree = degree # id kernel == 'poly'
		self.C = C
		self.coef0 = 0.0
		if self.kernel == 'poly' and self.coef0 <= 0.0:
			self.coef0 = 0.01
		else:
			self.coef0 = coef0

		self.verbose = verbose

		# generate regressor
		self.mdl = self.generate(kernel, 
			gamma, 
			#epsilon, 
			degree, 
			C,
			coef0,
			verbose = verbose)


	def generate(self, 
		kernel, gamma, #epsilon, 
		degree, C, coef0,
		verbose = True, from_own = True):
		"""
		"""

		if from_own:
			#if kernel == 'linear':
			#	svC = LinearSVC(
			#		C = self.C,
			#		verbose = self.verbose)
			#else:
			svC = SVC(kernel = self.kernel,
					gamma = self.gamma,
					#epsilon = self.epsilon,
					degree = self.degree,
					C = self.C,
					coef0 = self.coef0,
					verbose = self.verbose,
					#class_weight = 'balanced',
					probability = True)
		else:
			#if kernel == 'linear':
			#	svC = LinearSVC(
			#		C = C,
			#		verbose = verbose)
			#else:
			svC = SVC(kernel = kernel,
					gamma = gamma,
					#epsilon = epsilon,
					degree = degree,
					C = C,
					coef0 = coef0,
					verbose = verbose, 
					#class_weight = 'balanced',
					probability = True)	

		return svC


	def fit_model(self, 
		features, 
		labels, 
		opt = False, 
		num_cv = 3, 
		verbose = True,
		params = None):
		"""
		"""
		if opt:
			mdl = SVC()
			print ("num_cv", num_cv)
			self.mdl = self.hyperparam_optimise(features, 
				labels, 
				mdl,
				num_cv = num_cv)
		else:
			if params is not None:
				self.mdl = self.generate(params['kernel'],
					params['gamma'], 
					#params['epsilon'], 
					params['degree'],
					params['C'],
					params['coef0'],
					verbose = verbose,
					from_own = False)
				
			self.mdl.fit(features, labels)

		return self.mdl


	def hyperparam_optimise(self, 
		train_feature_arr, 
		train_label_arr, 
		model, 
		num_cv = 3):
		"""
		"""
		from sklearn.model_selection import GridSearchCV
		import numpy as np	
		from sklearn import metrics

		scoring_funcs = {'Brier':'brier_score_loss', 
			'AUC':'roc_auc',
			'F1':'f1'}
	
		param_grid = {
			'kernel':['linear', 'rbf', 'poly'], # for with fl
			'gamma':['scale', 'auto'],
			'C':[0.1, 1.0, 10, 100],
			'coef0':[0., 1.0]
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
			#if self.kernel != 'linear':
			predictions = self.mdl.predict_proba(features)
			#else:
			#	pass
		else:
			#if self.kernel != 'linear':
			predictions = self.mdl.predct_log_prob(features)
			#else:
			#	pass # Cali...?

		return predictions




		
