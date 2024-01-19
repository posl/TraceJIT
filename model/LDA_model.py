"""
LDA based classification model
"""

class LDA_model(object):
	"""docstring for LDA_model"""
	def __init__(self, n_components, 
		solver = 'eigen',
		tol = 0.0001):

		super(LDA_model, self).__init__()
		self.n_components = n_components
		self.solver = solver
		self.tol = tol
		self.mdl = self.generate()


	def generate(self):
		"""
		"""
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

		mdl = LDA(n_components = 1, 
			solver = self.solver,
			store_covariance = True, 
			tol = self.tol)

		return mdl


	def fit_model(self, features, labels, 
		opt = False, num_cv = 3, params = None):
		"""
		"""
		self.mdl.fit(features, labels)


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