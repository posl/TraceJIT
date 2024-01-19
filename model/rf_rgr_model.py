"""
DP model based on RandomForest
"""
GRIDPARAMS = {'RF':['max_depth', 'min_samples_split', 'n_estimators']}
from sklearn.ensemble import RandomForestRegressor

class RFR_model(object):
	"""docstring for RFR_model"""
	def __init__(self, n_estimators = 100, #100 
		max_depth = None, 
		min_samples_split = 2,
		random_state = None):

		super(RFR_model, self).__init__()
	
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.random_state = random_state

		# generate rgr
		self.mdl = self.generate(n_estimators, max_depth, min_samples_split, random_state)


	def generate(self, n_estimators, max_depth, min_samples_split, random_state, 
		from_own = True, 
		opt = False):
		"""
		"""
		if from_own:
			rgr = RandomForestRegressor(n_estimators = self.n_estimators, 
				max_depth = self.max_depth,
				min_samples_split = self.min_samples_split,
				random_state = self.random_state,
				verbose = 1)
		else:
			rgr = RandomForestRegressor(n_estimators = n_estimators, 
				max_depth = max_depth,
				min_samples_split = min_samples_split,
				random_state = random_state,
				verbose = 1)		

		return rgr


	def fit_model(self, features, labels, opt = False, num_cv = 3, params = None):
		"""
		"""
		if opt:
			mdl = RandomForestRegressor()
			print ("num_cv", num_cv)
			self.mdl = self.hyperparam_optimise(features, 
				labels, 
				mdl,
				num_cv = num_cv)

#			self.mdl = RandomForestRegressor(
#				n_estimators = best_params['n_estimators'],
#				max_depth = None,#best_params['max_depth'],
#				min_samples_split = best_params['min_samples_split'],
#				random_state = None,
#				)
		else:
			if params is not None:
				self.mdl = self.generate(params['n_estimators'], 
					params['max_depth'], 
					params['min_samples_split'], 
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

		scoring_funcs = {'Brier':'brier_score_loss', 
			'AUC':'roc_auc',
			'F1':'f1'}
	
		param_grid = {
			#'max_depth':list(np.int32(np.linspace(1,13,13))) + [None],
			'max_depth':list(np.int32(np.linspace(1,15,15))) + [None], # for with fl
			'min_samples_split':np.int32(np.linspace(2,10,5)),
			'n_estimators':[100,150,200],
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
			# thresholkd = 0.5
			ret_predictions = np.int32(predictions > 0.5)
		elif predtype == 'prob':
			ret_predictions = predictions
		else:
			ret_predictions = np.log(predictions)

		return ret_predictions

