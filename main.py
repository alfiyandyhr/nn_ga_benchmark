#NN-based surrogate optimization for benchmark problems
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/10/2021
#####################################################################################################
from LoadVars import *
from ga import *
from NeuralNet import *
from performance import *
import matplotlib.pyplot as plt

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
print(f'The benchmark problem: {problem_name.upper()}\n')

#####################################################################################################

#Initial sampling
initial_sampling = define_sampling(initial_sampling_method_name)
print(f'Performing initial sampling: {initial_sampling_method_name.upper()}\n')
InitialData = initial_sampling.do(problem, n_samples=pop_size, pop=None)
parent_pop = InitialData

#Evaluating initial samples (true eval)
InitialEval = problem.evaluate(InitialData, return_values_of=['F'])
InitialEval_G = problem.evaluate(InitialData, return_values_of=['G'])

if InitialEval_G is not None:
	InitialEval = np.concatenate((InitialEval, InitialEval_G), axis=1)

np.savetxt('DATA/training/X.dat',
	InitialData, delimiter=' ', newline='\n',
	footer="")

np.savetxt('DATA/training/OUT.dat',
	InitialEval, delimiter=' ', newline='\n',
	footer="")

#Initial performance
HV = [0]
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
									 return_least_infeasible=True)
print('Optimal solutions on the initial trained model is obtained!\n')
print('--------------------------------------------------')

#####################################################################################################

#Iterative trainings
for update in range(number_of_updates):
	#Saving best design variables (X_best) on every trained model
	print(f'Updating the training data to the neural net, update={update+1}\n\n')
	np.savetxt('DATA/prediction/X_best.dat',
		optimal_solutions.X, delimiter=' ', newline='\n',
		header='Row = sample point, column = design vars',
		footer="")
	
	with open('DATA/training/X.dat','a') as f:
		np.savetxt(f, optimal_solutions.X,
			delimiter=' ', newline='\n',
			footer="")
	
	#Evaluating X_best (true eval)
	Eval_X_best = problem.evaluate(optimal_solutions.X,
								   return_values_of=['F'])
	Eval_X_best_G = problem.evaluate(optimal_solutions.X,
							   	   return_values_of=['G'])
	if Eval_X_best_G is not None:
		Eval_X_best = np.concatenate((Eval_X_best, Eval_X_best_G), axis=1)

	with open('DATA/training/OUT.dat', 'a') as f:
		np.savetxt(f, Eval_X_best,
			delimiter=' ', newline='\n',
			footer="")

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
										 return_least_infeasible=True)
	print('--------------------------------------------------\n')
	print('Optimal solutions on the trained model is obtained!\n')
	print('--------------------------------------------------\n\n')

print(f'NN based surrogate optimization is DONE! True eval = {100*(number_of_updates+2)}\n')

#Evaluating the last X_best (true eval)
Eval_X_best = problem.evaluate(optimal_solutions.X,
							   return_values_of=['F'])


#Performance measurement for the last solutions
HV += [calc_hv(Eval_X_best[:,range(problem.n_obj)], ref=hv_ref)]

#Ideal performance (pareto front)
HV_pareto = calc_hv(problem.pareto_front(), ref=hv_ref)

#True evaluation counters
true_eval = [0]
for update in range(number_of_updates+2):
	true_eval += [n_gen*(update+1)]

#####################################################################################################

#Plot
if pf_plot or initial_samples_plot or optim_plot:
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

if hv_plot:
	plt.plot(true_eval, HV)
	plt.hlines(HV_pareto, 0, 2*len(true_eval)*100, colors='k', label='Pareto')
	plt.title(f'HV History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("HV value")
	plt.legend(loc="upper right")
	plt.show()

#####################################################################################################
