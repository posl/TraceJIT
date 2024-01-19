"""
DP models based on MLP
"""
import tensorflow as tf
from tensorflow.keras import layers
import random
import numpy as np


METRICS = [
	tf.keras.metrics.TruePositives(name='tp'),
	tf.keras.metrics.FalsePositives(name='fp'),
	tf.keras.metrics.TrueNegatives(name='tn'),
	tf.keras.metrics.FalseNegatives(name='fn'), 
	tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	tf.keras.metrics.Precision(name='precision'),
	tf.keras.metrics.Recall(name='recall'),
	tf.keras.metrics.Recall(name='f1'),
	tf.keras.metrics.AUC(name='auc')
]


class MLP_model(object):
	"""docstring for MLP_model"""
	def __init__(self, 
		num_features,
		hidden_layer_sizes = [100],
		dropouts = [None],
		batch_size = 256,
		epochs = 200,
		learning_rate = 1e-4,
		random_state = None):

		super(MLP_model, self).__init__()
		self.hidden_layer_sizes = hidden_layer_sizes
		self.dropouts = dropouts
		self.batch_size = batch_size
		self.epochs = epochs
		self.learning_rate = learning_rate 
		
		#self.random_state = random_stat
		random.seed(random_state)

		# generate rgr
		self.mdl = self.generate(self.hidden_layer_sizes,
			self.dropouts,
			self.learning_rate,
			num_features)


	def generate(self, 
		hidden_layer_sizes,
		dropouts, 
		learning_rate,
		num_features):
		"""
		"""
		model = tf.keras.Sequential()

		model.add(layers.Conv1D(filters = 32, 
			kernel_size = 3, strides = 1, 
			activation = 'relu', padding="valid",
			input_shape = (num_features,1)))#, padding="same"))
		
		#model.add(layers.Dropout(0.3))
		model.add(layers.MaxPooling1D(pool_size = 2, padding="valid"))
		
		model.add(layers.Conv1D(filters = 32, 
			kernel_size = 3, strides = 1, 
			activation = 'relu', padding="valid"))#, padding="same"))
		#model.add(layers.Dropout(0.2))
		model.add(layers.MaxPooling1D(pool_size = 2, padding="valid"))

		model.add(layers.Flatten())
		for layer_size, dropout_v in zip(hidden_layer_sizes, dropouts):
			model.add(layers.Dense(layer_size, activation = 'relu'))
			if dropout_v is not None:
				model.add(layers.Dropout(dropout_v))

		# last
		model.add(layers.Dense(1, activation = 'sigmoid'))

		model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate),
			loss = 'binary_crossentropy',#'mean_squared_error', # binary_crossentropy
			metrics = METRICS)# METRICS)

		return model
#		if from_own:
#			clf = MLPClassifier(
#				hidden_layer_sizes = hidden_layer_sizes,
#				solver = self.solver,
#				max_iter = self.max_iter,
#				learning_rate_init = self.learning_rate_init,
#				learning_rate = self.learning_rate,
#				random_state = self.random_state,
#				verbose = 1)
#		else:
#			clf = MLPClassifier(
#				hidden_layer_sizes = hidden_layer_sizes,
#				solver = solver,
#				max_iter = max_iter,
#				learning_rate_init = learning_rate_init,
#				learning_rate = learning_rate,
#				random_state = random_state,
#				verbose = 1)#

#		return clf


	def fit_model(self, features, labels, epochs = 0, batch_size = 0):
		"""
		"""
		if batch_size <= 0:
			batch_size = self.batch_size
		
		if epochs <= 0:
			epochs = self.epochs

		print ('Feature', features.shape)
		print ("labels", labels.shape)
		print ("batch_size", batch_size)

		self.mdl.fit(features.reshape(features.shape[0],features.shape[1],1), 
			labels, 
			epochs = epochs, 
			batch_size = batch_size,
			validation_split=0.33)

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
		batch_size = 0,
		predtype = 'label'):
		"""
		predtype: label, prob, log_prob
		"""

		if batch_size <= 0:
			batch_size = self.batch_size

		if predtype == 'label':
			predictions = self.mdl.predict(features.reshape(features.shape[0],features.shape[1],1), 
				batch_size = batch_size).reshape(-1)
			#predictions = [np.round(pred_v) for pred_v in predictions]
			predictions = np.int32(predictions > 0.5)
		elif predtype == 'prob':
			#print ("HERE")
			predictions = self.mdl.predict(features.reshape(features.shape[0],features.shape[1],1),
				batch_size = batch_size).reshape(-1)

		return predictions

