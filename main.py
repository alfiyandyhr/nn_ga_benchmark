#NN-based surrogate optimization for benchmark problems
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/12/2021
#####################################################################################################
from LoadVars import *
from ga import *
from NeuralNet import NeuralNet, train, calculate
from performance import calc_hv
from SaveOutput import save
import matplotlib.pyplot as plt
import torch

import os
#####################################################################################################

print('------------------------------------------------------')
print('--- NN based surrogate optimization coded by Alfi ----')
print('------------------------------------------------------\n')
print('Successfully loaded input data, now initializing...\n')

#####################################################################################################

#Defining problem
problem = define_problem(problem_name)
pareto_front = problem.pareto_front()
save('OUTPUT/pareto_front.dat', problem.pareto_front(), header=f'Pareto Front of {problem_name}')
print(f'The benchmark problem: {problem_name.upper()}\n')

#####################################################################################################

#Initial sampling
initial_sampling = define_sampling(initial_sampling_method_name)
print(f'Performing initial sampling: {initial_sampling_method_name.upper()}\n')
InitialData = initial_sampling.do(problem, n_samples=pop_size, pop=None)

#Evaluating initial samples (true eval)
InitialEval = problem.evaluate(InitialData, return_values_of=['F'])
InitialEval_G = problem.evaluate(InitialData, return_values_of=['G'])
InitialEval_CV = problem.evaluate(InitialData, return_values_of=['CV'])

if InitialEval_G is not None:
	InitialEval = np.concatenate((InitialEval, InitialEval_G, InitialEval_CV), axis=1)

save('OUTPUT/initial_pop_X.dat', InitialData, header='Generation 1: X')
save('OUTPUT/initial_pop_FGCV.dat', InitialEval, header='Generation 1: F, G, CV')
save('OUTPUT/all_pop_X.dat', InitialData, header='Generation 1: X')
save('OUTPUT/all_pop_FGCV.dat', InitialEval, header='Generation 1: F, G, CV')
save('DATA/training/X.dat', InitialData)
save('DATA/training/OUT.dat', InitialEval[:, 0:problem.n_obj+problem.n_constr])

#Initial performance
HV = [0.0]
HV += [calc_hv(InitialEval[:,range(problem.n_obj)], ref=hv_ref)]

#####################################################################################################

#Initial training for neural nets
print('Feeding the training data to the neural net...\n\n')

Model = NeuralNet(D_in=problem.n_var,
				  H=N_Neuron, D=N_Neuron,
				  D_out=problem.n_obj+problem.n_constr)

print('Performing initial training...\n')

TrainedModel = train(problem=problem,
					 model=Model,
				     N_Epoch=N_Epoch,
				     lr=lr,
				     batchrate=batchrate)

print('\nAn initial trained model is obtained!\n')
print('--------------------------------------------------')
TrainedModel_Problem = TrainedModelProblem(problem, TrainedModel)

#####################################################################################################

#Evolutionary computation routines on the Trained Model
selection = define_selection(selection_operator_name)
crossover = define_crossover(crossover_operator_name, prob=prob_c, eta=eta_c)
mutation = define_mutation(mutation_operator_name, eta=eta_m)

#EA settings
EA = EvolutionaryAlgorithm(algorithm_name)
algorithm = EA.setup(pop_size=pop_size,
					 sampling=initial_sampling,
					 # selection=selection,
					 crossover=crossover,
					 mutation=mutation)

#Stopping criteria
stopping_criteria_def = StoppingCriteria(termination_name)
stopping_criteria = stopping_criteria_def.set_termination(n_gen=n_gen)

#Obtaining optimal solutions on the initial trained model
print(f'Performing optimization on the initial trained model using {algorithm_name.upper()}\n')
optimal_solutions =  do_optimization(TrainedModel_Problem,
									 algorithm, stopping_criteria,
									 verbose=True, seed=1,
									 return_least_infeasible=False)
print('--------------------------------------------------')
print('\nOptimal solutions on the initial trained model is obtained!\n')
print('--------------------------------------------------')

#####################################################################################################

#Iterative trainings
for update in range(number_of_updates):
	#Saving best design variables (X_best) on every trained model
	print(f'Updating the training data to the neural net, update={update+1}\n\n')
	with open('DATA/prediction/X_best.dat','a') as f:
		save(f, optimal_solutions.X, header=f'Generation {update+2}: X')

	with open('OUTPUT/all_pop_X.dat', 'a') as f:
		save(f, optimal_solutions.X, header=f'Generation {update+2}: X') 
	
	with open('DATA/training/X.dat','a') as f:
		save(f, optimal_solutions.X)
	
	#Evaluating X_best (true eval)
	Eval_X_best = problem.evaluate(optimal_solutions.X,
								   return_values_of=['F'])
	Eval_X_best_G = problem.evaluate(optimal_solutions.X,
							   	     return_values_of=['G'])
	Eval_X_best_CV = problem.evaluate(optimal_solutions.X,
									  return_values_of=['CV'])

	if Eval_X_best_G is not None:
		Eval_X_best = np.concatenate((Eval_X_best, Eval_X_best_G, Eval_X_best_CV), axis=1)

	with open('OUTPUT/all_pop_FGCV.dat', 'a') as f:
		save(f, Eval_X_best, header=f'Generation {update+2}: F, G, CV')

	with open('DATA/training/OUT.dat', 'a') as f:
		save(f, Eval_X_best[:, 0:problem.n_obj+problem.n_constr]) 

	#Performance measurement for each iteration
	HV += [calc_hv(Eval_X_best[:,range(problem.n_obj)], ref=hv_ref)]

	#Training neural nets
	print(f'Performing neural nets training, training={update+2}\n')

	TrainedModel = train(problem=problem,
						 model=TrainedModel,
						 N_Epoch=N_Epoch,
						 lr=lr,
						 batchrate=batchrate)

	TrainedModel_Problem = TrainedModelProblem(problem, TrainedModel)

	#Optimal solutions
	print('--------------------------------------------------\n')
	print(f'Performing optimization on the trained model using {algorithm_name.upper()}\n')
	optimal_solutions =  do_optimization(TrainedModel_Problem,
										 algorithm, stopping_criteria,
										 verbose=True, seed=1,
										 return_least_infeasible=False)
	print('--------------------------------------------------\n')
	print('Optimal solutions on the trained model is obtained!\n')
	print('--------------------------------------------------\n\n')

#Evaluating the last X_best (true eval)
Eval_X_best = problem.evaluate(optimal_solutions.X,
							   return_values_of=['F'])
Eval_X_best_G = problem.evaluate(optimal_solutions.X,
							   	 return_values_of=['G'])
Eval_X_best_CV = problem.evaluate(optimal_solutions.X,
								  return_values_of=['CV'])

if Eval_X_best_G is not None:
	Eval_X_best = np.concatenate((Eval_X_best, Eval_X_best_G, Eval_X_best_CV), axis=1)

save('OUTPUT/final_pop_X.dat', optimal_solutions.X, header=f'Generation {number_of_updates+2}: X')
save('OUTPUT/final_pop_FGCV.dat', Eval_X_best, header=f'Generation {number_of_updates+2}: F, G, CV')
with open('OUTPUT/all_pop_X.dat', 'a') as f:
	save(f, optimal_solutions.X, header=f'Generation {number_of_updates+2}: X')
with open('OUTPUT/all_pop_FGCV.dat', 'a') as f:
	save(f, Eval_X_best, header=f'Generation {number_of_updates+2}: F, G, CV') 

#Performance measurement for the last solutions
HV += [calc_hv(Eval_X_best[:,range(problem.n_obj)], ref=hv_ref)]

#Ideal performance (pareto front)
HV_pareto = [calc_hv(problem.pareto_front(), ref=hv_ref)]

#True evaluation counters
true_eval = [0]
for update in range(number_of_updates+2):
	true_eval += [pop_size*(update+1)]

true_eval = np.array([true_eval]).T
HV = np.array([HV]).T
HV_pareto = np.array(HV_pareto)
HV = np.concatenate((HV, true_eval),axis=1)

save('OUTPUT/HV.dat', HV, header='HV History: HV value, true eval counters')
save('OUTPUT/HV_pareto.dat', HV_pareto, header=f'HV pareto of {problem_name.upper()}')

print(f'NN based surrogate optimization is DONE! True eval = {100*(number_of_updates+2)}\n')
#####################################################################################################
