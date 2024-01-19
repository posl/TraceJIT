"""
Support Vector Regression (SVR) model
"""
from sklearn.svm import SVR

class SVR_model(object):
	"""docstring for SVR_model"""
	def __init__(self, kernel = 'rbf', 
		gamma = 'scale', 
		epsilon = 0.1, 
		degree = 3, 
		C = 1.0,
		coef0 = 1, 
		verbose = True):

		super(SVR_model, self).__init__()

		self.kernel = kernel
		self.gamma = gamma
		self.epsilon = epsilon
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
			epsilon, 
			degree, 
			C,
			coef0,
			verbose = verbose)


	def generate(self, 
		kernel, gamma, epsilon, degree, C, coef0,
		verbose = True, from_own = True):
		"""
		"""

		if from_own:
			svr = SVR(kernel = self.kernel,
				gamma = self.gamma,
				epsilon = self.epsilon,
				degree = self.degree,
				C = self.C,
				coef0 = self.coef0,
				verbose = self.verbose)
		else:
			svr = SVR(kernel = kernel,
				gamma = gamma,
				epsilon = epsilon,
				degree = degree,
				C = C,
				coef0 = coef0,
				verbose = verbose)	

		return svr


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
			mdl = SVR()
			print ("num_cv", num_cv)
			self.mdl = self.hyperparam_optimise(features, 
				labels, 
				mdl,
				num_cv = num_cv)
		else:
			if params is not None:
				self.mdl = self.generate(params['kernel'],
					params['gamma'], 
					params['epsilon'], 
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
		import numpy as np

		predictions = np.float16(self.mdl.predict(features))
		if predtype == 'label':
			ret_predictions = np.int32(predictions > 0.5)
		elif predtype == 'prob':
			ret_predictions = np.asarray([min(max(pred_prob, 0.0), 1.0) for pred_prob in predictions])
		else:
			ret_predictions = np.log(predictions)

		return ret_predictions








		
