#Genetic Algorithms using pymoo
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/8/2021
#####################################################################################################
import numpy as np

from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_sampling, get_selection
from pymoo.factory import get_crossover, get_mutation, get_termination
from pymoo.visualization.scatter import Scatter

from NeuralNet import calculate
#####################################################################################################
#Disable warning
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

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
	def __init__(self, problem, model):
		"""Inheritance from Problem class"""
		self.n_var = problem.n_var
		self.n_obj = problem.n_obj
		self.n_constr = problem.n_constr
		self.xl = problem.xl
		self.xu = problem.xu
		self.problem = problem
		self.model = model
		super().__init__(n_var=self.n_var,
						 n_obj=self.n_obj,
						 n_constr=self.n_constr,
						 xl=self.xl, xu=self.xu)
	
	def _evaluate(self, X, out, *args, **kwargs):
		"""Evaluation method"""
		OUT = calculate(X=X,
					    problem=self.problem,
					    model=self.model)

		F = OUT[:, 0:self.n_obj]
		G = OUT[:, self.n_obj:(self.n_obj+self.n_constr)]

		out["F"] = np.column_stack([F])
		out["G"] = np.column_stack([G])

class ProblemBenchmark():
	"""Instance for benchmarks"""
	def __init__(self, name):
		"""Name of the benchmark"""
		self.name = name

	def get_problem(self):
		"""Returning the python object"""
		problem = get_problem(self.name)
		return problem

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

class SamplingDefinition():
	"""Instance for the sampling"""
	def __init__(self, name):
		"""Name of the sampling method"""
		self.name = name
	def get_sampling(self):
		"""Returning the python object"""	
		sampling_method = get_sampling(self.name)
		return sampling_method

class SelectionOperator():
	"""Instance for the selection operator"""
	def __init__(self, name):
		"""Name of the selection operator"""
		self.name = name
	def get_selection(self):
		"""Returning the python object"""	
		if self.name == 'tournament':
			selection = get_selection(self.name,
				func_comp='real_tournament')
		return selection

class CrossoverOperator():
	"""Instance for the crossover operator"""
	def __init__(self, name):
		"""Name of the crossover operator"""
		self.name = name
	def get_crossover(self, prob, eta):
		"""Returning the python object"""	
		crossover = get_crossover(self.name, prob=prob, eta=eta)
		return crossover

class MutationOperator():
	"""Instance for the mutation operator"""
	def __init__(self, name):
		"""Name of the mutation operator"""
		self.name = name
	def get_mutation(self, eta):
		"""Returning the python object"""	
		mutation = get_mutation(self.name, eta=eta)
		return mutation

class StoppingCriteria():
	"""Instance for the termination"""
	def __init__(self, name):
		"""Name of the stopping criteria"""
		self.name = name
	def get_termination(self, n_gen):
		"""Returning the python object"""	
		termination = get_termination(self.name, n_gen)
		return termination

def do_optimization(problem, algorithm, termination,
	verbose=False, seed=1, return_least_infeasible=True):
	"""Conduct optimization process and return optimized solutions"""
	optim = minimize(problem, algorithm, termination,
					 verbose=verbose, seed=seed,
					 return_least_infeasible=return_least_infeasible)
	return optim
