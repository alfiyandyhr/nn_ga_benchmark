#NN-based surrogate optimization for benchmark problems
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/8/2021
#####################################################################################################
from LoadVars import *
from ga import *
import matplotlib.pyplot as plt
from NeuralNet import *

import os
#####################################################################################################

print('------------------------------------------------------')
print('--- NN based surrogate optimization coded by Alfi ----')
print('------------------------------------------------------\n')
print('Successfully loaded input data, now initializing...\n')

#####################################################################################################

#Defining problem
problem_def = ProblemBenchmark(problem_name)
problem = problem_def.get_problem()
pareto_front = problem.pareto_front()
print(f'The benchmark problem: {problem_name.upper()}\n')

#####################################################################################################

#Initial sampling
initial_sampling_def = SamplingDefinition(initial_sampling_method)
initial_sampling = initial_sampling_def.get_sampling()
print(f'Performing initial sampling: {initial_sampling_def.name.upper()}\n')
InitialData = initial_sampling.do(problem, n_samples=pop_size, pop=None)

#Evaluating initial samples (true eval)
InitialEval = problem.evaluate(InitialData, return_values_of=['F'])
InitialEval_G = problem.evaluate(InitialData, return_values_of=['G'])

if InitialEval_G is not None:
	InitialEval = np.concatenate((InitialEval, InitialEval_G), axis=1)

np.savetxt('DATA/training/X_init.dat',
	InitialData, delimiter=' ', newline='\n',
	header='Row = sample point, column = design vars',
	footer="")

np.savetxt('DATA/training/OUT_init.dat',
	InitialEval, delimiter=' ', newline='\n',
	header='Row = sample point, column = objective and constraint (if any)',
	footer="")

#####################################################################################################

#Initial training for neural nets
print('Feeding the training data to the neural net...\n\n')
Model = NeuralNet(D_in=problem.n_var,
				  H=N_Neuron, D=N_Neuron,
				  D_out=problem.n_obj+problem.n_constr)
print('Performing initial training...\n')
TrainedModel = train(model=Model, N_Epoch=N_Epoch, lr=lr, init=True)
print('\nAn initial trained model is obtained!\n')
print('--------------------------------------------------')
TrainedModel_Problem = TrainedModelProblem(problem_name, TrainedModel)

#####################################################################################################

#Evolutionary computation routines on the Trained Model
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

#Obtaining optimal solutions on the initial trained model
print(f'Performing optimization on the initial trained model using {algorithm_name.upper()}\n')
optimal_solutions =  do_optimization(TrainedModel_Problem,
									 algorithm, stopping_criteria,
									 verbose=True, seed=1)
print('Optimal solutions on the initial trained model is obtained!\n')
print('--------------------------------------------------')

# print(optimal_solutions.F)
#####################################################################################################

#Iterative trainings
for update in range(number_of_updates-1):
	#Saving best design variables (X_best) on every trained model
	print(f'Updating the training data to the neural net, update={update+2}\n\n')
	np.savetxt('DATA/prediction/X_best.dat',
		optimal_solutions.X, delimiter=' ', newline='\n',
		header='Row = sample point, column = design vars',
		footer="")
	np.savetxt('DATA/training/X.dat',
		optimal_solutions.X, delimiter=' ', newline='\n',
		header='Row = sample point, column = design vars',
		footer="")
	
	#Evaluating X_best (true eval)
	Eval_X_best = problem.evaluate(optimal_solutions.X,
								   return_values_of=['F'])
	Eval_X_best_G = problem.evaluate(optimal_solutions.X,
							   	   return_values_of=['G'])
	if Eval_X_best_G is not None:
		Eval_X_best = np.concatenate((Eval_X_best, Eval_X_best_G), axis=1)


	np.savetxt('DATA/training/OUT.dat',
		Eval_X_best, delimiter=' ', newline='\n',
		header='Row = sample point, column = objective and constraint',
		footer="")

	#Training neural nets
	Model = TrainedModel
	print(f'Performing neural nets training, training={update+2}\n')
	TrainedModel = train(model=Model, N_Epoch=N_Epoch, lr=lr, init=False)
	TrainedModel_Problem = TrainedModelProblem(problem_name, TrainedModel)

	#Optimal solutions
	print(f'Performing optimization on the trained model using {algorithm_name.upper()}\n')
	optimal_solutions =  do_optimization(TrainedModel_Problem,
										 algorithm, stopping_criteria,
										 verbose=True, seed=1)
	print('--------------------------------------------------\n')
	print('Optimal solutions on the trained model is obtained!\n')
	print('--------------------------------------------------\n\n')

print(f'NN based surrogate optimization is DONE! True eval = {100*number_of_updates}\n')

#Evaluating the last X_best (true eval)
Eval_X_best = problem.evaluate(optimal_solutions.X,
							   return_values_of=['F'])

#####################################################################################################

#Plot

if pf_plot:
	plt.plot(pareto_front[:,0],pareto_front[:,1],'k-', label='Pareto front')
if initial_samples_plot:
	plt.plot(InitialEval[:,0], InitialEval[:,1], 'bo', label='Initial samples')
if optim_plot:
	plt.plot(Eval_X_best[:,0], Eval_X_best[:,1], 'ro', label='Optimal solutions')
# plt.plot(optimal_solutions.F[:,0],optimal_solutions.F[:,1],'ro') 
plt.title(f'Objective functions space of {problem_name.upper()}')
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend(loc="upper right")
plt.show()

#####################################################################################################
