#Evaluator
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/15/2021
#####################################################################################################
from pymoo.core.evaluator import Evaluator
import numpy as np

def evaluate(problem, pop):
	"""
	This function will do the true evaluation and return the array
	which contains F, G and CV
	Input:
		problem = problem object in pymoo
		pop = population object in pymoo
	Output:
		array of F, G and CV
	"""
	Evaluator().eval(problem, pop)

	pop_eval = pop.get('F')
	pop_G = pop.get('G')
	pop_CV = pop.get('CV')


	if pop_G[0] is not None:
		pop_eval = np.concatenate((pop_eval, pop_G, pop_CV), axis=1)
	else:
		pop_G = np.zeros((len(pop_eval),1))
		pop_eval = np.concatenate((pop_eval, pop_G, pop_CV), axis=1)

	return pop_eval