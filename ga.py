#Genetic Algorithms using pymoo
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/13/2021
#####################################################################################################
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.core.sampling import Sampling
from pymoo.core.evaluator import Evaluator
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.factory import get_problem
from pymoo.termination import get_termination

from NeuralNet import calculate
#####################################################################################################
#Disable warning
# from pymoo.configuration import Configuration
# Configuration.show_compile_hint = False

class UserDefinedProblem(Problem):
	"""A custom problem defined by users"""
	def __init__(self):
		"""Inheritance from Problem class"""
		super().__init__(n_var=2, n_obj=2, n_constr=2,
			xl=np.array([-2,-2]), xu=np.array([2,2]),
			elementwise_evaluation=True)

	def _evaluate(self, x, out, *args, **kwargs):
		"""Evaluation method"""
		f1 = x[0]**2 + x[1]**2
		f2 = (x[0]-1)**2 + x[1]**2

		g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
		g2 = -20*(x[0]-0.4) * (x[0]-0.6) / 4.8

		out["F"] = np.column_stack([f1, f2])
		out["G"] = np.column_stack([g1, g2])

class TrainedModelProblem(Problem):
	"""This is the trained neural net model"""
	def __init__(self, problem, device):
		"""Inheritance from Problem class"""
		self.n_var = problem.n_var
		self.n_obj = problem.n_obj
		self.n_eq_constr = problem.n_eq_constr
		self.n_ieq_constr = problem.n_ieq_constr
		self.xl = problem.xl
		self.xu = problem.xu
		self.problem = problem
		self.device = device
		
		super().__init__(n_var=self.n_var,
						 n_obj=self.n_obj,
						 n_constr=self.n_constr,
						 xl=self.xl, xu=self.xu)
	
	def _evaluate(self, X, out, *args, **kwargs):
		"""Evaluation method"""
		OUT = calculate(X=X, problem=self.problem,
						device=self.device)

		F = OUT[:, 0:self.n_obj]
		G = OUT[:, self.n_obj:(self.n_obj+self.n_constr)]

		out["F"] = np.column_stack([F])
		out["G"] = np.column_stack([G])

def define_problem(name):
		"""Returning the python object of the benchmark problem"""
		return get_problem(name)

class EvolutionaryAlgorithm():
	"""Instance for the crossover operator"""
	def __init__(self, name):
		"""Name of the crossover operator"""
		self.name = name
	def setup(self, pop_size, sampling,
			 crossover, mutation):
		#"""Returning the python object"""
		if self.name == 'nsga2':	
			algorithm = NSGA2(pop_size=pop_size,
							  # selection=selection,
							  sampling=sampling,
							  crossover=crossover,
							  mutation=mutation)
		else:
			print('Please enter the algorithm name!\n')

		return algorithm

def define_sampling(name):
		"""Returning the python object of the initial sampling method"""	
		return get_sampling(name)

def define_selection(name):
		"""Returning the python object of selection operator"""	
		if name == 'tournament':
			selection = TournamentSelection
		return selection

def define_crossover(name, prob, eta):
		"""Returning the python of the crossover operator"""
		if name == 'sbx':
			crossover = SBX(prob=prob, eta=eta)
		return crossover

def define_mutation(name, eta):
		"""Returning the python object of the mutation operator"""
		if name == 'pm':
			mutation = PolynomialMutation(eta=eta)	
		return mutation

class StoppingCriteria():
	"""Instance for the termination"""
	def __init__(self, name):
		"""Name of the stopping criteria"""
		self.name = name
	def set_termination(self, n_gen):
		"""Returning the python object"""	
		termination = get_termination(self.name, n_gen)
		return termination

def set_individual(X, F, G, CV):
	"""
	This will return an individual class in pymoo
	"""
	return Individual(X=X, F=F, G=G, CV=CV)

def set_population(n_individuals):
	"""
	This will return a population class in pymoo
	"""
	return Population(n_individuals=n_individuals)

def do_survival(problem, merged_pop, n_survive):
	"""This will merge two pops and return the best surviving pop"""
	Survival = RankAndCrowdingSurvival()
	surviving_pop = Survival.do(problem=problem, pop=merged_pop, n_survive=n_survive)

	surviving_pop_eval = surviving_pop.get('F')
	surviving_pop_G = surviving_pop.get('G')
	surviving_pop_CV = surviving_pop.get('CV')

	if surviving_pop_G[0] is not None:
		surviving_pop_eval = np.concatenate((surviving_pop_eval, surviving_pop_G, surviving_pop_CV), axis=1)
	else:
		surviving_pop_G = np.zeros((len(surviving_pop_eval),1))
		surviving_pop_eval = np.concatenate((surviving_pop_eval, surviving_pop_G, surviving_pop_CV), axis=1)

	return surviving_pop, surviving_pop_eval

def do_optimization(problem, algorithm, termination,
	verbose=False, seed=1, return_least_infeasible=True):
	"""Conduct optimization process and return optimized solutions"""
	optim = minimize(problem, algorithm, termination,
					 verbose=verbose, seed=seed,
					 return_least_infeasible=return_least_infeasible)
	return optim
