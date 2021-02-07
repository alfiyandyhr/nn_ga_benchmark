#NN-based surrogate optimization for benchmark problems
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/6/2021
#####################################################################################################
from LoadVars import *
from ga import *
import matplotlib.pyplot as plt
from NeuralNet import *

import os
#####################################################################################################

#Create folders for storing data
# os.makedirs('DATA')
# os.makedirs('DATA/prediction')
# os.makedirs('DATA/training')
# os.makedirs('DATA/storage')

print('Successfully loaded input data, now initializing...\n')

#Defining problem
problem_def = ProblemBenchmark(problem)
problem = problem_def.get_problem()
pareto_front = problem.pareto_front()
print(f'The benchmark problem: {problem_def.name.upper()}')

#Initial sampling
initial_sampling_def = SamplingDefinition(initial_sampling_method)
initial_sampling = initial_sampling_def.get_sampling()
print(f'Initial sampling using {initial_sampling_def.name.upper()}')
InitialData = initial_sampling.do(problem, n_samples=pop_size, pop=None)

#Evaluating initial samples (true eval)
InitialEval = problem.evaluate(InitialData, return_values_of=['F'])

np.savetxt('DATA/training/design_vars_init.dat',
	InitialData, delimiter=' ', newline='\n',
	header='Row = sample point, column = design vars',
	footer="")

np.savetxt('DATA/training/objs_constrs_init.dat',
	InitialEval, delimiter=' ', newline='\n',
	header='Row = sample point, column = objective functions and constraint functions',
	footer="")

#Training neural nets
# Model = NeuralNet(D_in=30, H=100,
# 				  D=100, D_out=2)

# TrainedModel = train(model=Model, N_Epoch=1000)

#Evolutionary computation routines on the Trained Model

#Selecting genetic operators for EA setup
selection_def = SelectionOperator(selection_operator_name)
#selection = selection_def.get_selection()
crossover_def = CrossoverOperator(crossover_operator_name)
crossover = crossover_def.get_crossover(prob=prob_c, eta=eta_c)
mutation_def = MutationOperator(mutation_operator_name)
mutation = mutation_def.get_mutation(eta=eta_m)

#Evolutionary algorithms
EA = EvolutionaryAlgorithm(algorithm_name)
algorithm = EA.setup(pop_size=pop_size,
					 sampling=initial_sampling,
					 # selection=selection,
					 crossover=crossover,
					 mutation=mutation)

#Stopping criteria
stopping_criteria_def = StoppingCriteria(termination_name)
stopping_criteria = stopping_criteria_def.get_termination(n_gen=n_gen)

#Optimal solutions
optimal_solutions =  do_optimization(problem, algorithm, stopping_criteria,
	verbose=True, seed=1)

# plt.plot(InitialEval[:,0], InitialEval[:,1], 'o')
plt.plot(pareto_front[:,0],pareto_front[:,1])
# plt.plot(optimal_solutions.F[:,0],optimal_solutions.F[:,1],'ro')
plt.show()