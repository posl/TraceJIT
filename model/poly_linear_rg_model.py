"""
Logistic Regression model
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegression_model(object):
	"""docstring for PolynomialRegression_model"""

	def __init__(self, 
		fit_intercept,
		degree = 2):
		super(PolynomialRegression_model, self).__init__()

		self.degree = degree
		self.interaction_only = False
		self.include_bias = True
		self.order = 'C'

		self.mdl = self.generate(fit_intercept = fit_intercept)


	def generate(self,
		fit_intercept = True,
		degree = 2):
		"""
		Paramters:
			fit_intercept=True, normalize=False, copy_X=True, n_jobs=None
		"""

		clf = LogisticRegression(
			fit_intercept = True,
			normalize = False,
			copy_X = True,
			n_jobs = None)

		return clf


	def fit_model(self, 
		features, 
		labels, 
		opt = False, 
		num_cv = 3, 
		params = None):
		"""
		"""
		#if opt:
		#	model = LogisticRegression()
		#	print ("num_cv", num_cv)
		#	self.mdl = self.hyperparam_optimise(features, 
		#		labels, 
		#		model, 
		#		num_cv = num_cv)
		#else:
		#	if params is not None:
		#		self.mdl = self.generate(self.random_state, 
		#			params['penalty'], 
		#			params['solver'], 
		#			True, 
		#			params['C'],
		#			from_own = False)
		if opt:
			pass
			#model = PolynomialFeatures()
			#self.mdl = self.hyperparam_optimise(features,
			#	labels,
			#	model,
			#	num_cv = num_cv)
		else:
			#degree=2, interaction_only=False, include_bias=True, order='C'
			if params is not None:
				features = PolynomialFeatures(
					degree = params['degree'],
					interaction_only = params['interaction_only'],
					include_bias = params['include_bias'],
					order = params['order']).fit_transform(features)
			else:
				features = PolynomialFeatures(
					degree = self.degree,
					interaction_only = self.interaction_only,
					include_bias = self.include_bias,
					order = self.order).fit_transform(features)				

			self.mdl.fit(features, labels)

		return self.mdl

#	def hyperparam_optimise(self, train_feature_arr, train_label_arr, model, num_cv = 3):
#		"""
#		"""
#		from sklearn.model_selection import GridSearchCV
#		import numpy as np	
#		from sklearn import metrics#
#		scoring_funcs = {'Brier':'brier_score_loss', 
#			'AUC':'roc_auc',
#			'F1':'f1'}
#	
#		param_grid = {
#			'degree':[2,3,4]
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
#		#best_params = grid_search.best_params_
#		print ("BEST PARAMS", grid_search.best_params_)
#		#sys.exit()
#		return grid_search


	def predict(self, features, predtype = 'label'):
		"""
		predtype: label, prob, log_prob
		"""

		features = PolynomialFeatures(
			degree = self.degree,
			interaction_only = self.interaction_only,
			include_bias = self.include_bias,
			order = self.order).fit_transform(features)

		if predtype == 'label':
			predictions = self.mdl.predict(features)
		elif predtype == 'prob':
			#print ("HERE")
			predictions = self.mdl.predict_proba(features)
		else:
			predictions = self.mdl.predct_log_prob(features)

		return predictions



