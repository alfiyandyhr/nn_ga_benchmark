#NN-based surrogate optimization for benchmark problems
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/12/2021
#####################################################################################################
from LoadVars import *
from performance import calc_hv, calc_igd
from SaveOutput import save
import numpy as np
import matplotlib.pyplot as plt
import torch

if use_nn:
	from ga import *
	from NeuralNet import NeuralNet, train, calculate
	from eval import evaluate

if not use_nn:
	from pymoo.algorithms.nsga2 import NSGA2
	from pymoo.factory import get_problem, get_sampling
	from pymoo.factory import get_selection, get_crossover, get_mutation
	from pymoo.factory import get_termination
	from pymoo.optimize import minimize
	import copy

#Perform cuda computation if NVidia GPU card available
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

#####################################################################################################
if use_nn:
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
	parent_pop = initial_sampling.do(problem, n_samples=pop_size)

	#Evaluating initial samples (true eval)
	parent_pop_eval = evaluate(problem, parent_pop)

	save('OUTPUT/initial_pop_X.dat', parent_pop.get('X'), header='Generation 1: X')
	save('OUTPUT/initial_pop_FGCV.dat', parent_pop_eval, header='Generation 1: F, G, CV')
	save('OUTPUT/all_pop_X.dat', parent_pop.get('X'), header='Generation 1: X')
	save('OUTPUT/all_pop_FGCV.dat', parent_pop_eval, header='Generation 1: F, G, CV')
	save('DATA/training/X.dat', parent_pop.get('X'))
	save('DATA/training/OUT.dat', parent_pop_eval[:, 0:problem.n_obj+problem.n_constr])

	#Initial performance
	HV = [0.0]
	HV += [calc_hv(parent_pop_eval[:,range(problem.n_obj)], ref=hv_ref)]
	IGD = [1E-5]
	IGD += [calc_igd(parent_pop_eval[:,range(problem.n_obj)], pareto_front)]

	####################################################################################################

	#Initial training for neural nets
	print('Feeding the training data to the neural net...\n\n')

	Model = NeuralNet(D_in=problem.n_var,
					  H=N_Neuron, D=N_Neuron,
					  D_out=problem.n_obj+problem.n_constr)

	print('Performing initial training...\n')

	train(problem=problem,
		  model=Model,
	      N_Epoch=N_Epoch,
	      lr=lr,
	      train_ratio=train_ratio,
	      batchrate=batchrate,
	      device=device)

	print('\nAn initial trained model is obtained!\n')
	print('--------------------------------------------------')

	TrainedModel_Problem = TrainedModelProblem(problem, device)

	#####################################################################################################

	#Evolutionary computation routines on the Trained Model
	selection = define_selection(selection_operator_name)
	crossover = define_crossover(crossover_operator_name, prob=prob_c, eta=eta_c)
	mutation = define_mutation(mutation_operator_name, eta=eta_m)
	Survival = RankAndCrowdingSurvival()

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
	res =  do_optimization(TrainedModel_Problem,
						   algorithm, stopping_criteria,
						   verbose=True, seed=1,
						   return_least_infeasible=True)
	print('--------------------------------------------------')
	print('\nOptimal solutions on the initial trained model is obtained!\n')
	print('--------------------------------------------------')

	#####################################################################################################

	#Iterative trainings
	for update in range(number_of_updates):
		#Saving best design variables (X_best) on every trained model
		print(f'Updating the training data to the neural net, update={update+1}\n\n')
		with open('DATA/prediction/X_best.dat','a') as f:
			save(f, res.X, header=f'Generation {update+2}: X') 
		
		with open('DATA/training/X.dat','a') as f:
			save(f, res.X)
		
		#Evaluating X_best (true eval)
		child_pop = pop_from_array_or_individual(res.X)
		child_pop_eval = evaluate(problem, child_pop)

		with open('DATA/training/OUT.dat', 'a') as f:
			save(f, child_pop_eval[:, 0:problem.n_obj+problem.n_constr]) 

		#Merging parent_pop and child_pop
		merged_pop = Population.merge(parent_pop, child_pop)

		#Survival method depending on the algorithm type
		parent_pop, parent_pop_eval = do_survival(problem, merged_pop, n_survive=pop_size)

		with open('OUTPUT/all_pop_X.dat', 'a') as f:
			save(f, parent_pop.get('X'), header=f'Generation {update+2}: X')

		with open('OUTPUT/all_pop_FGCV.dat', 'a') as f:
			save(f, parent_pop_eval, header=f'Generation {update+2}: F, G, CV')

		#Performance measurement for each iteration
		HV  += [calc_hv(parent_pop.get('F')[:,range(problem.n_obj)], ref=hv_ref)]
		IGD += [calc_igd(parent_pop.get('F')[:,range(problem.n_obj)], pareto_front)]

		#Training neural nets
		print(f'Performing neural nets training, training={update+2}\n')

		Model = torch.load('DATA/prediction/trained_model.pth').to(device)

		train(problem=problem,
			  model=Model,
			  N_Epoch=N_Epoch,
			  lr=lr,
			  train_ratio=train_ratio,
			  batchrate=batchrate,
			  device=device)

		Model = torch.load('DATA/prediction/trained_model.pth').to(device)
		TrainedModel_Problem = TrainedModelProblem(problem, device)

		#Optimal solutions
		print('--------------------------------------------------\n')
		print(f'Performing optimization on the trained model using {algorithm_name.upper()}\n')
		res =  do_optimization(TrainedModel_Problem,
							   algorithm, stopping_criteria,
							   verbose=True, seed=1,
							   return_least_infeasible=True)
		print('--------------------------------------------------\n')
		print('Optimal solutions on the trained model is obtained!\n')
		print('--------------------------------------------------\n\n')

	#Evaluating the last X_best (true eval)
	child_pop = pop_from_array_or_individual(res.X)
	child_pop_eval = evaluate(problem, child_pop)

	#Merging parent_pop and child_pop
	merged_pop = Population.merge(parent_pop, child_pop)

	#Survival method depending on the algorithm type
	parent_pop, parent_pop_eval = do_survival(problem, merged_pop, n_survive=pop_size)

	save('OUTPUT/final_pop_X.dat', parent_pop.get('X'), header=f'Generation {number_of_updates+2}: X')
	save('OUTPUT/final_pop_FGCV.dat', parent_pop_eval, header=f'Generation {number_of_updates+2}: F, G, CV')
	with open('OUTPUT/all_pop_X.dat', 'a') as f:
		save(f, parent_pop.get('X'), header=f'Generation {number_of_updates+2}: X')
	with open('OUTPUT/all_pop_FGCV.dat', 'a') as f:
		save(f, parent_pop_eval, header=f'Generation {number_of_updates+2}: F, G, CV') 

	#Performance measurement for the last solutions
	HV  += [calc_hv(parent_pop_eval[:,range(problem.n_obj)], ref=hv_ref)]
	IGD += [calc_igd(parent_pop_eval[:,range(problem.n_obj)], pareto_front)]

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
	IGD[0] = IGD[1]
	IGD = np.array([IGD]).T
	IGD = np.concatenate((IGD, true_eval),axis=1)

	save('OUTPUT/HV.dat', HV, header='HV History: HV value, true eval counters')
	save('OUTPUT/HV_pareto.dat', HV_pareto, header=f'HV pareto of {problem_name.upper()}')
	save('OUTPUT/IGD.dat', IGD, header='IGD History: IGD value, true eval counters')

	print(f'NN based surrogate optimization is DONE! True eval = {pop_size*(number_of_updates+2)}\n')
	#####################################################################################################

if not use_nn:
	print('-------------------------------------------------------------------')
	print('--- Multi-objective optimization using pymoo by Julian and Deb ----')
	print('-------------------------------------------------------------------\n')
	print('Successfully loaded input data, now initializing...\n')

	#####################################################################################################

	#Defining problem
	problem = get_problem(problem_name)
	pareto_front = problem.pareto_front()
	save('OUTPUT/PURE_GA/pareto_front.dat', problem.pareto_front(), header=f'Pareto Front of {problem_name}')
	print(f'The benchmark problem: {problem_name.upper()}\n')

	#####################################################################################################

	#Evolutionary computation routines
	sampling = get_sampling(initial_sampling_method_name)
	# selection = get_selection(selection_operator_name_ga)
	crossover = get_crossover(crossover_operator_name_ga, prob=prob_c_ga, eta=eta_c_ga)
	mutation = get_mutation(mutation_operator_name_ga, eta=eta_m_ga)

	#EA settings
	if algorithm_name == 'nsga2':
		algorithm = NSGA2(pop_size=pop_size,
						  sampling=sampling,
						  # selection=selection,
						  crossover=crossover,
						  mutation=mutation)

	#Stopping criteria
	termination = get_termination(termination_name, n_gen_ga)

	#Optimization process (all true evaluations)
	print(f'Performing optimization on the {problem_name.upper()} using {algorithm_name.upper()}\n')
	
	obj = copy.deepcopy(algorithm)

	obj.setup(problem, termination, seed=1)

	HV  = [0.0]
	IGD = [1E-5]

	while obj.has_next():
		obj.next()

		pop_eval = obj.pop.get('F')
		pop_G = obj.pop.get('G')
		pop_CV = obj.pop.get('CV')

		if pop_G[0] is not None:
			pop_eval = np.concatenate((pop_eval, pop_G, pop_CV), axis=1)
		else:
			pop_G = np.zeros((len(pop_eval),1))
			pop_eval = np.concatenate((pop_eval, pop_G, pop_CV), axis=1)

		if obj.n_gen == 1:
			save('OUTPUT/PURE_GA/initial_pop_X.dat', obj.pop.get('X'), header='Generation 1: X')
			save('OUTPUT/PURE_GA/initial_pop_FGCV.dat', pop_eval, header='Generation 1: F, G, CV')
			save('OUTPUT/PURE_GA/all_pop_X.dat', obj.pop.get('X'), header='Generation 1: X')
			save('OUTPUT/PURE_GA/all_pop_FGCV.dat', pop_eval, header='Generation 1: F, G, CV')

		if obj.n_gen != 1:
			with open('OUTPUT/PURE_GA/all_pop_X.dat', 'a') as f:
				save(f, obj.pop.get('X'), header=f'Generation {obj.n_gen}: X')
			with open('OUTPUT/PURE_GA/all_pop_FGCV.dat', 'a') as f:
				save(f, pop_eval, header=f'Generation {obj.n_gen}: F, G, CV')

		if obj.n_gen == n_gen_ga:
			save('OUTPUT/PURE_GA/final_pop_X.dat', obj.pop.get('X'), header=f'Generation {n_gen_ga}: X')
			save('OUTPUT/PURE_GA/final_pop_FGCV.dat', pop_eval, header=f'Generation {n_gen_ga}: F, G, CV')

		#Performance measurement for every generation
		HV  += [calc_hv(pop_eval[:,range(problem.n_obj)], ref=hv_ref)]
		IGD += [calc_igd(pop_eval[:,range(problem.n_obj)], pareto_front)]

		print(f"Generation = {obj.n_gen}; n_nds = {len(obj.opt)}; HV = {HV[obj.n_gen]}; IGD = {IGD[obj.n_gen]}")


	#Ideal performance (pareto front)
	HV_pareto = [calc_hv(problem.pareto_front(), ref=hv_ref)]

	#True evaluation counters
	true_eval = [0]
	for update in range(n_gen_ga):
		true_eval += [pop_size*(update+1)]

	true_eval = np.array([true_eval]).T
	HV  = np.array([HV]).T
	HV_pareto = np.array(HV_pareto)
	HV  = np.concatenate((HV, true_eval),axis=1)
	IGD[0] = IGD[1]
	IGD = np.array([IGD]).T
	IGD = np.concatenate((IGD, true_eval),axis=1)

	save('OUTPUT/PURE_GA/HV.dat', HV, header='HV History: HV value, true eval counters')
	save('OUTPUT/PURE_GA/HV_pareto.dat', HV_pareto, header=f'HV pareto of {problem_name.upper()}')
	save('OUTPUT/PURE_GA/IGD.dat', IGD, header='IGD History: IGD value, true eval counters')

	print('--------------------------------------------------')
	print('\nOptimal solutions are obtained!\n')
	print('--------------------------------------------------')

	# #Saving output
	# n_evals = []
	# F, G, CV = [], [], []

	# for algorithm in res.history:
	# 	n_evals.append(algorithm.evaluator.n_eval)

	# 	opt = algorithm.opt

	# 	F.append(opt.get('F'))
	# 	G.append(opt.get('G'))
	# 	CV.append(opt.get('CV'))

	# HV = []

	# for gen in range(len(F)):
	# 	HV.append(calc_hv(F[gen], hv_ref))

	# HV_pareto = calc_hv(problem.pareto_front(),hv_ref)

	# print(CV[0])

	# plt.plot(F[0][:,0],F[0][:,1],'o')
	# plt.show()

	# plt.plot(n_evals, HV)
	# plt.hlines(HV_pareto,0,n_evals[len(n_evals)-1],colors='k')
	# plt.show()