"""
DP models based on MLP
"""
import sys
from deap import base, creator, tools, algorithms, gp
from tqdm import tqdm
import numpy as np
import operator
import random

def protect_div(x, y):
	"""
	"""
	if y != 0:
		return x/y
	else:
		return 1

def protect_sqrt(x):
	"""
	"""
	return np.sqrt(np.abs(x))


def if_then_else(input, 
	output1, 
	output2):
	return output1 if input else output2


def draw(individual, destfile):
	"""
	Draw a graph for a given individual
	"""
	nodes, edges, labels = gp.graph(individual)	

	### Graphviz Section ###
	import pygraphviz as pgv	

	g = pgv.AGraph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	g.layout(prog="dot")	

	for i in nodes:
		n = g.get_node(i)
		n.attr["label"] = labels[i]	

	g.draw(destfile)


class GP_model(object):
	"""docstring for MLP_model"""
	def __init__(self, 
		headers,
		maxTreeDepth = 8, 
		minTreeDepth = 1, 
		initMaxTreeDepth = 6, 
		cxpb = 0.9,
		mutpb = 0.1, 
		random_state = None,
		num_pop = 40,
		ngen = 100,
		is_pareto = False,
		use_dist_as_fit = False,
		use_min_max_scaler = False):

		super(GP_model, self).__init__()

		self.random_state = random_state
		self.headers = headers 
		self.maxTreeDepth = maxTreeDepth
		self.minTreeDepth = minTreeDepth
		self.initMaxTreeDepth = initMaxTreeDepth

		self.ngen = ngen
		self.num_pop = num_pop
		self.is_pareto = is_pareto
		self.use_dist_as_fit = use_dist_as_fit

		random.seed(self.random_state)
		np.random.seed(self.random_state)

		self.toolbox = self.generate(
			self.headers,
			self.maxTreeDepth, 
			self.minTreeDepth, 
			self.initMaxTreeDepth, 
			num_pop = self.num_pop)

		self.cxpb = cxpb
		self.mutpb = mutpb

		self.mdl = None
		self.threshold = 0.

		# for dist
		self.opt_prob_dist = None
		self.use_min_max_scaler = use_min_max_scaler


	def predict_probas(self,
		features,
		individual):
		"""
		"""
		pred_probas = np.zeros(features.shape[0], dtype = np.float32)

		individual_in_exec_form = individual
		for idx_to_feature, feature_name in enumerate(self.headers):
			individual_in_exec_form = individual_in_exec_form.replace(feature_name, 
				"features[i,{}]".format(idx_to_feature))

		#compute suspicousness score for each data row(per method) in current_data
		for i in range(features.shape[0]):
			pred_probas[i] = eval(individual_in_exec_form)

		#print (np.min(features), np.max(features))
		#print (np.max(pred_probas), np.min(pred_probas), np.median(pred_probas), np.mean(pred_probas), np.std(pred_probas))

		return pred_probas

	def normalise_lil(self, vs):
		"""
		"""
		new_vs = np.zeros(vs.shape)
		new_vs[:] = vs
#		if np.min(new_vs) == 0.:
#			#print (new_vs)
#			if len(new_vs[new_vs > 0.]) > 0:
#				eps = np.min([np.min(new_vs[new_vs > 0.]), np.finfo(np.float32).eps])
#			else:
#				eps = np.finfo(np.float32).eps#

#			new_vs[new_vs == 0.] = eps
		# normalise then to make their sum to 1.
		normed = new_vs/np.sum(new_vs)
		if np.min(normed) == 0.:
			#print (normed)
			if len(normed[normed > 0.]) > 0: # have values larger than 0.
				eps = np.min([np.min(normed[normed > 0.]), np.finfo(np.float32).eps])
			else:
				eps = np.finfo(np.float32).eps

			normed[normed == 0.] = eps

		return normed


	def set_prob_for_ground_truth(self):
		"""
		"""
		eps = np.finfo(np.float32).eps

#		opt_prob_dist = np.zeros(self.labels.shape, dtype = np.float32)
#		opt_prob_dist[self.labels == 0] = eps
#		cnt_class_1 = np.sum(self.labels)
#		opt_prob_dist[self.labels == 1] = np.float32(1./cnt_class_1)
#		# scale 
#		opt_prob_dist = self.scale(opt_prob_dist)
		opt_prob_dist = np.zeros(self.labels.shape, dtype = np.float32)
		opt_prob_dist[self.labels == 0] = 0.
		cnt_class_1 = np.sum(self.labels)
		opt_prob_dist[self.labels == 1] = 0.5 + 0.5*np.float32(1./cnt_class_1)

		self.opt_prob_dist = self.normalise_lil(self.scale(opt_prob_dist))


	def lil_score(self, 
		pred_probas):
		"""
		"""
		if self.opt_prob_dist is None:
			self.set_prob_for_ground_truth()

		processed_probs = self.normalise_lil(self.scale(pred_probas))
		lil = np.sum(self.opt_prob_dist * np.log(self.opt_prob_dist/processed_probs))

		return lil


	def eval_func(self, individual):
		"""
		Evalauet current solutioin 
		"""
		pred_probas = self.predict_probas(self.features, str(individual))
		predictions = 1 * (pred_probas > self.threshold) #self.threshold)	
		if self.use_dist_as_fit:
			from sklearn.metrics import f1_score
			assert self.is_pareto and self.use_dist_as_fit, "{} and {}".format(self.is_pareto, self.use_dist_as_fit)

			from scipy.spatial.distance import pdist
			in_class_1 = pred_probas[self.labels == 1]
			in_class_0 = pred_probas[self.labels == 0]

			assert len(in_class_1)>0 and len(in_class_0)>0, "At least have one datapoint for each class: {}(1),{}(0)".format(
				len(in_class_1), len(in_class_0))

			#avg_dist_in_class_1 = np.mean(pdist(in_class_1.reshape(-1,1)))
			avg_dist_btwn_class_1 = np.std(in_class_1)
			#avg_dist_in_class_0 = np.mean(pdist(in_class_0.reshape(-1,1)))
			avg_dist_btwn_class_0 = np.std(in_class_0)
			avg_dist_within_classes = np.mean([avg_dist_in_class_1, avg_dist_btwn_class_1])
			#lil_score_v = -self.lil_score(pred_probas)

			dist_from_thrshold_class_1 = np.mean(in_class_1) - self.threshold
			dist_from_thrshold_class_0 = self.threshold - np.mean(in_class_0)
			avg_dist_from_threshold = np.mean([dist_from_thrshold_class_1, dist_from_thrshold_class_0]) # similar to macro f1 score

			f1 = np.float32(f1_score(self.labels, predictions, average = 'binary'))
			#fitness = (-np.float32(avg_dist_in_class_1),-np.float32(avg_dist_in_class_0), np.float32(dist_btwn_class_1_0))
			#fitness = (-np.float32(avg_dist_in_class_1),-np.float32(avg_dist_in_class_0),avg_dist_from_threshold)
			fitness = (lil_score_v, f1)
		else:
			from sklearn.metrics import f1_score, accuracy_score, roc_auc_score#, log_loss #confustion_matrix, f1_score	
			#print (pred_probas)
			#print (predictions)
			#num_class_1 = np.sum(self.labels)
			#num_class_0 = len(self.labels) - num_class_1
			# F1
			#tn, fp, fn, tp  = confustion_matrix(self.labels, 
			#	predictions, 
			#	labels = [0,1])
			if self.is_pareto:
				f1 = np.float32(f1_score(self.labels, predictions, average = 'binary')) # macro, binary, micro, weighted 
				#acc = np.float32(accuracy_score(self.labels, predictions))
				pred_probas = self.scale(pred_probas)
				#print (pred_probas)
				auc = np.float32(roc_auc_score(self.labels, pred_probas))
				fitness = (f1, auc)
			else:
				#fitness_2 = f1_score(self.labels, predictions, average = 'micro') # micro, weighted 
				#fitness_3 = f1_score(self.labels, predictions, average = 'weighted') # micro, weighted 
				#fitness_4 = f1_score(self.labels, predictions, average = 'binary') # micro, weighted 		
				#entropy_loss = log_loss(self.labels, predictions)
				#print ("Fitness", fitness_1, fitness_2, fitness_3, fitness_4, entropy_loss)
				### just testing 
				#fitness = np.mean(np.asarray(self.labels) * 100 - pred_probas)
				f1 = np.float32(f1_score(self.labels, predictions, average = 'binary')) # macro, binary, micro, weighted 
				#pred_probas = self.scale(pred_probas)
				#auc = np.float32(roc_auc_score(self.labels, pred_probas))
				#acc = np.float32(accuracy_score(self.labels, predictions))
				fitness = (f1,)
			#print (fitness)
			#sys.exit()
		return fitness
		

	def selTournament_mo_vers(self,
		individuals, 
		#tournament_size, 
		k):
		"""
		return winners of k number of tournament.
		without duplicate 
		"""
		num_ind = len(individuals)
		tournsize = int(num_ind * 0.2) if int(num_ind * 0.2) % 2 == 0 else int(num_ind * 0.2) + 1
		assert tournsize >= k, "{}(tourn) should be larger or eqaul to {}(k)".format(
			tournsize, k)
		
		winners = []
		remained_individuals = [self.toolbox.clone(ind) for ind in individuals]
		remained_individuals_in_str = [str(ind) for ind in individuals]
		#while len(winners) < k:
		for _ in range(k):
			sampled_inds = tools.selRandom(remained_individuals, tournsize)
			selected = tools.selNSGA2(sampled_inds, 1)[0]
			#print (selected)
			remained_individuals.pop(remained_individuals_in_str.index(str(selected)))
			remained_individuals_in_str.pop(remained_individuals_in_str.index(str(selected)))
			#print ("--", len(individuals), len(remained_individuals))
			winners.append(selected)

			#if len(winners) >= k:
			#	break
		#sys.exit()
		#if len(winners) > k:
		#	from operator import attrgetter
		#	sorted_front = sorted(winners, key=attrgetter("fitness.crowding_dist"), reverse=True)
		#	return sorted_front[:k]
		#else:
		return winners

	def gen_primitives(self, headers):
		"""
		"""
		pset = gp.PrimitiveSet('main', len(headers)) # arity = len(headers)

		pset.addPrimitive(np.add, 2, name = "np.add")
		pset.addPrimitive(np.subtract, 2, name = "np.subtract")
		pset.addPrimitive(np.multiply, 2, name = "np.multiply")
		pset.addPrimitive(np.negative, 1, name = "np.negative")
		pset.addPrimitive(protect_div, 2, name = "protect_div")
		pset.addPrimitive(protect_sqrt, 1, name = "protect_sqrt")
		#pset.addPrimitive(if_then_else, [bool, float, float], float)

		pset.addTerminal(1.0) # vary ..?
		#pset.addEphemeralConstant("C", lambda: np.float32(np.random.uniform(0., 100.))) # generate betwen 0 to 1

		# rename
		for idx, feature_name in enumerate(headers):
			pset.renameArguments(**{"ARG" + str(idx):feature_name})
			print ("ARG{} -> {}".format(idx, feature_name))

		#sys.exit()
		return pset


	def generate(self,
		headers, 
		maxTreeDepth, 
		minTreeDepth, 
		initMaxTreeDepth, 
		num_pop = 40):
		"""
		"""
		
		#Setting parameter(feature) for evolving: pset == feature set
		pset = self.gen_primitives(headers)

		if ("Fitness" in globals()):
			del creator.Fitness

		if not self.is_pareto:
			creator.create("Fitness", base.Fitness, weights = (1.0,))
		elif not self.use_dist_as_fit:
			creator.create("Fitness", base.Fitness, weights = (1.0,1.0))
		else:
			creator.create("Fitness", base.Fitness, weights = (-1.0,1.0))#,1.0))

		if ("Individual" in globals()):
			del creator.Individual

		creator.create("Individual", gp.PrimitiveTree, 
			fitness = creator.Fitness, 
			pset = pset)	

		toolbox = base.Toolbox()	

		toolbox.register("expr", gp.genHalfAndHalf, pset = pset, min_ = minTreeDepth, max_ = initMaxTreeDepth)
		toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = 2)	

		toolbox.register("compile", gp.compile, pset = pset)
		
		toolbox.register("evaluate", self.eval_func)	

		# This is for parent selection, not the survivor selection
		#selBest is for the elitism
		tournsize = int(num_pop * 0.2) if int(num_pop * 0.2) % 2 == 0 else int(num_pop * 0.2) + 1
		if not self.is_pareto:
			toolbox.register("select", tools.selTournament, tournsize = tournsize, fit_attr = "fitness")
		else:
			toolbox.register("select", self.selTournament_mo_vers)#tools.selTournamentDCD)

		#toolbox.register("selectNSGA", tools.selNSGA2)
		
		#parents = tools.selTournament(pop, 2, tournsize, fit_attr = 'fitness')
		toolbox.register("mate", gp.cxOnePoint)
		toolbox.register("mutate", gp.mutUniform, expr = toolbox.expr, pset = pset)	

		toolbox.decorate("mate", gp.staticLimit(key = operator.attrgetter("height"), max_value = maxTreeDepth))
		toolbox.decorate("mutate", gp.staticLimit(key = operator.attrgetter("height"), max_value = maxTreeDepth))	

		self.toolbox = toolbox

		return toolbox


	def check_duplicate(self, 
		offspring, 
		selected):
		"""
		Return True if it is duplicated, else return False
		"""
		is_duplicated = str(offspring) == str(selected)	
		return is_duplicated


	def run(self,
		features, 
		labels,
		ngen,
		num_pop,
		cxpb,
		mutpb,
		toolbox,
		num_best = 1):	
		"""
		main method where genetic programming is actually taken place
		"""
		# set feature and labels
		self.features = features #/ 100
		self.labels = np.asarray(labels)

		pop = toolbox.population(n = num_pop)
		# statistics
		if not self.is_pareto:
			stats = tools.Statistics(lambda ind: ind.fitness.values[0])
		else:
			stats = tools.Statistics(lambda ind: ind.fitness.values)

		stats.register("average", np.mean, axis = 0)
		stats.register("max", np.max, axis = 0)
		stats.register("min", np.min, axis = 0)

		fitness_per_ind = toolbox.map(toolbox.evaluate, pop)
		for fit, ind in zip(fitness_per_ind, pop):
			ind.fitness.values = fit
			print ("-\t", str(ind), fit)
			#print (str(ind), fit)
		#sys.exit()
		if self.is_pareto:
			tools.emo.assignCrowdingDist(pop)

		# set current best
		if not self.is_pareto:
			best = tools.HallOfFame(num_best, similar = self.check_duplicate)
		else:
			best = tools.ParetoFront(similar = self.check_duplicate)

		best.update(pop)	

		stats_result = stats.compile(pop)
		prev_max_fit = stats_result['max']

		# main algo
		for idx_to_gen in tqdm(range(1, ngen + 1)):

			# use tournament selection or rolutee-wheel selection
			next_pop = []	

			#if not self.is_pareto:
			while len(next_pop) < num_pop:# select a pair of individuals that will become the parent
				parents = toolbox.select(pop, 2) # use fitness of the individual for the tournament
				print ("Parent_1", str(parents[0]))
				print ("Parent_2", str(parents[1]) + "\n")
				#parents = random.sample(pop, 2)
				assert len(parents) == 2, len(parents)

				# generate a single offspring
				#offsprings = algorithms.varOr(parents, toolbox, 1, cxpb, mutpb)		
				offsprings = algorithms.varAnd(parents, toolbox, cxpb, mutpb)	
				for offspring in offsprings:	
					if len(next_pop) == 0 or not self.check_duplicate(offspring, next_pop):
						next_pop.append(offspring)
			#else:
			#	cand_offsprings = toolbox.select(pop, num_pop)
			#	for ind1, ind2 in zip(cand_offsprings[::2], cand_offsprings[1::2]):
			#		offsprings = algorithms.varOr([ind1, ind2], toolbox, 1, cxpb, mutpb)
			#		next_pop.extend(offsprings)
			#
			#	print (len(next_pop), num_pop, len(cand_offsprings))
			#sys.exit()

			#re-evaluate fitness for the offspring
			#generate evaluation data	
			num_offsprings = len(next_pop)
			assert num_offsprings == num_pop, "{} vs {}".format(num_offsprings, num_pop)
			####

			fitness_per_ind = toolbox.map(toolbox.evaluate, next_pop)	
			#update next_pop, with invalid fitness, with the new(valid) fitness
			for fit, ind in zip(fitness_per_ind, next_pop): 
				ind.fitness.values = fit
				print ("-\t", str(ind), fit)
			# select next population
			# Survival Selection -> select the ones go into the next generation
			if not self.is_pareto:
				pop[:]= tools.selBest(pop + next_pop, num_pop)
			else:
				pop[:] = tools.selNSGA2(pop + next_pop, num_pop)

			#update current best for this new pop -> HallOfFame
			best.update(pop)

			#Logging current status the pop
			stats_result = stats.compile(pop)
			#print "=================Statistics================="
			
			if self.use_dist_as_fit:
				curr_pareto_front = tools.sortNondominated(pop, num_pop, first_front_only = True)
				#logging
				print ("\t\tGeneration " + str(idx_to_gen) + " - number of individuals in pareto front: " + str(len(curr_pareto_front)))

				stat_summary = "\t\tGeneration {}: ".format(idx_to_gen)
				max_stat = ""; min_stat = ""; mean_stat = ""
				for idx,eval_metric in enumerate(['Dist_cls1', 'Dist_cls0']):#, 'Dist_btwn_cls1_0']):
					max_stat += "{0:.4f}({0:.4f}),".format(stats_result["max"][idx], eval_metric)
					min_stat += "{0:.4f}({0:.4f}),".format(stats_result["min"][idx], eval_metric)
					mean_stat += "{0:.4f}({0:.4f}),".format(stats_result["average"][idx], eval_metric)

				max_stat = max_stat[:-1]
				min_stat = min_stat[:-1]
				mean_stat = mean_stat[:-1]
				
				stat_summary = "\t\tGeneration {}: {}(Max), {}(AVG), {}(MIN)".format(
					idx_to_gen, max_stat, min_stat, mean_stat)
				print (stat_summary)
			elif self.is_pareto: # f1 and accuracy
				# for logging
				#curr_pareto_front = tools.sortNondominated(pop, num_pop, first_front_only = True)
				print ("ParetoFront for Gen {}".format(idx_to_gen))
				cnt = 0
				for i_to_cand,cand in enumerate(best):
					print ("\nCand {}: {}".format(i_to_cand, str(cand)))
					print ("\t\t fitness: {}".format(",".join(["{0:.4f}".format(v) for v in cand.fitness.values])))	
					cnt += 1

				print ("Total {} number of candidate in pareto front".format(cnt))

				stat_summary = "\t\tGeneration {}: ".format(idx_to_gen)
				max_stat = ""; min_stat = ""; mean_stat = ""
				for idx,eval_metric in enumerate(['f1', 'accuracy']):
					max_stat += "{0:.4f}({0:.4f}),".format(stats_result["max"][idx], eval_metric)
					min_stat += "{0:.4f}({0:.4f}),".format(stats_result["min"][idx], eval_metric)
					mean_stat += "{0:.4f}({0:.4f}),".format(stats_result["average"][idx], eval_metric)

				max_stat = max_stat[:-1]
				min_stat = min_stat[:-1]
				mean_stat = mean_stat[:-1]

				stat_summary = "\t\tGeneration {}: {}(Max), {}(AVG), {}(MIN)".format(
					idx_to_gen, max_stat, min_stat, mean_stat)
				print (stat_summary)
			else: # only f1 as fitness
				print ("\t\tGeneration {}: {}(Max), {}(AVG), {}(MIN)".format(
					idx_to_gen, stats_result["max"], stats_result["average"], stats_result["min"]))	

		return best


	def fit_model(self,
		features,
		labels, 
		cxpb = None,# = 0.9,
		mutpb = None,# = 0.1, 
		num_pop = None,# = 40, 
		ngen = None,# = 100,
		num_of_best = 1,
		params = None):
		"""
		"""

		if cxpb is None:
			cxpb = self.cxpb

		if mutpb is None:
			mutpb = self.mutpb

		if num_pop is None:
			num_pop = self.num_pop

		if ngen is None:
			ngen = self.ngen


		if self.toolbox is None:
			if self.params is not None:
				self.toolbox = self.generate(
					params['headers'],
					params['maxTreeDepth'], 
					params['minTreeDepth'], 
					params['initMaxTreeDepth'], 
					num_pop = params['num_pop'])
			else:
				self.toolbox = self.generate(
					self.headers,
					self.maxTreeDepth, 
					self.minTreeDepth, 
					self.initMaxTreeDepth, 
					num_pop = num_pop)

		# run
		best = self.run(
			features, 
			labels,
			ngen,
			num_pop,
			cxpb,
			mutpb,
			self.toolbox,
			num_best = 1)

		if not self.is_pareto:
			print ("\nThe best individual ")
			print ("\t\t expression : " + str(best[0]))
			print ("\t\t fitness   : " + str(best[0].fitness.values[0]))
			self.mdl = best[0]
		else: 
			print ("Candidate:")
			cnt = 0
			for i,cand in enumerate(best):
				print ("\nCand {}: {}".format(i, str(cand)))
				print ("\t\t fitness   : {}".format(",".join([str(v) for v in cand.fitness.values])))	
				cnt += 1
			print ("Total {} number of candidate".format(cnt))

			the_best = tools.selNSGA2(best, 1)[0]
			print ("\nThe best individual ")
			print ("\t\t expression : " + str(the_best))
			print ("\t\t fitness   : {}".format(", ".join([str(v) for v in the_best.fitness.values])))			
			self.mdl = the_best

		return self.mdl


	def scale(self, vs):
		"""
		"""
		if self.use_min_max_scaler:
			return self.min_max_scale(vs)
		else:
			from sklearn.preprocessing import MinMaxScaler
			min_v = np.min(vs)
			max_v = np.max(vs)	

			dist_to_left = self.threshold - min_v 
			dist_to_right = max_v - self.threshold 
			
			indices_to_l = [i for i,v in enumerate(vs) if v <= self.threshold]
			vs_in_l = np.asarray(vs)[indices_to_l]
			indices_to_r = [i for i,v in enumerate(vs) if v > self.threshold]
			vs_in_r = np.asarray(vs)[indices_to_r]	

			# combine indices of left side and right side into one (in order)
			indices = indices_to_l + indices_to_r	

			if dist_to_left == 0 and dist_to_right == 0:
				return np.full((len(vs),), 0.5)
			elif dist_to_right >= dist_to_left:
				right_side = [0.5, 1.0]
				left_side = [0.5 - 0.5 * dist_to_left/dist_to_right, 0.5]
			else: # dist_to_right < dist_to_left
				left_side = [0., 0.5]
				right_side = [0.5, 0.5 + 0.5 * dist_to_right/dist_to_left]	

			print  ('Left and Right', left_side, right_side)
			print ("Number in left side and right side: {} and {}".format(len(vs_in_l), len(vs_in_r)))
			if len(vs_in_l) > 0:
				scaler_l = MinMaxScaler()
				scaled_vs_in_l = (left_side[1]-left_side[0])*scaler_l.fit_transform(vs_in_l.reshape(-1,1)).reshape(-1,) + left_side[0]
				#print (scaled_vs_in_l.shape)
			else:
				scaled_vs_in_l = np.empty(0,)	

			if len(vs_in_r) > 0:
				scaler_r = MinMaxScaler()
				scaled_vs_in_r = (right_side[1]-right_side[0])*scaler_r.fit_transform(vs_in_r.reshape(-1,1)).reshape(-1,) + right_side[0]
				#print (scaled_vs_in_r.shape)
			else:
				scaled_vs_in_r = np.empty(0,)	

			scaled_vs_not_yet_in_order = np.concatenate((scaled_vs_in_l, scaled_vs_in_r), axis = 0)	

			scaled = np.zeros(len(indices))
			for idx, scaled_v in enumerate(scaled_vs_not_yet_in_order):
				scaled[indices[idx]] = scaled_v
			
			#print ("======= ", scaled)
			#from sklearn.preprocessing import MaxAbsScaler
			#scaler = MaxAbsScaler()
			#scaled = scaler.fit_transform(vs.reshape(-1,1)).reshape(-1,)
			#print ("======= ", scaled)
			return scaled.clip(0.,1.)
			#return scaled.clip(0., 1.)


	def min_max_scale(self, vs):
		"""
		Here, instead of a complex scaling function, we will simply
		use min-max scaler, converting vs within 0 to 1.
		Thus, self.threshold will be set to 0.5, no matter of its initial value
		"""
		from sklearn.preprocessing import MinMaxScaler

		scaler = MinMaxScaler()
		scaled = scaler.fit_transform(np.asarray(vs).reshape(-1,1)).flatten()

		if self.threshold != 0.5:
			self.threshold = 0.5

		return scaled


	def predict(self, 
		features, 
		predtype = 'label'):
		"""
		predtype: label, prob, log_prob
		"""

		if predtype == 'label':
			predictions = 1 * (self.predict_probas(features, str(self.mdl)) > self.threshold)
		elif predtype == 'prob':
			from sklearn.preprocessing import MinMaxScaler
			predictions = self.predict_probas(features, str(self.mdl))
			#print (predictions.shape)
			#scaler = MinMaxScaler()
			#predictions = scaler.fit_transform(predictions.reshape(-1,1)).reshape(-1,).clip(0.0, 1.0)
			predictions = self.scale(predictions)
			#print ("in prob", np.max(predictions), np.min(predictions))
		else:
			predictions = np.log(self.predict_probas(features, str(self.mdl)))
			
		return predictions




