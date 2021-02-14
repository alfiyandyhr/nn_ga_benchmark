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

#Initial sampling and evaluations
initial_sampling = define_sampling(initial_sampling_method_name)
print(f'Performing initial sampling: {initial_sampling_method_name.upper()}\n')
parent_pop = initial_sampling.do(problem, n_samples=pop_size)
Evaluator().eval(problem, parent_pop)

print(parent_pop.get('CV', 'G')[0])

# #Initial performance
# HV = [0.0]
# HV += [calc_hv(InitialEval[:,range(problem.n_obj)], ref=hv_ref)]

# print(HV)
# #####################################################################################################
# #Initial training for neural nets
# print('Feeding the training data to the neural net...\n\n')

# Model = NeuralNet(D_in=problem.n_var,
# 				  H=N_Neuron, D=N_Neuron,
# 				  D_out=problem.n_obj+problem.n_constr)

# print('Performing initial training...\n')

# TrainedModel = train(problem=problem,
# 					 model=Model,
# 				     N_Epoch=N_Epoch,
# 				     lr=lr,
# 				     batchrate=batchrate)

# print('\nAn initial trained model is obtained!\n')
# print('--------------------------------------------------')
# TrainedModel_Problem = TrainedModelProblem(problem, TrainedModel)

# #####################################################################################################

# #Evolutionary computation routines on the Trained Model
# selection = define_selection(selection_operator_name)
# crossover = define_crossover(crossover_operator_name, prob=prob_c, eta=eta_c)
# mutation = define_mutation(mutation_operator_name, eta=eta_m)

# #EA settings
# EA = EvolutionaryAlgorithm(algorithm_name)
# algorithm = EA.setup(pop_size=pop_size,
# 					 sampling=initial_sampling,
# 					 # selection=selection,
# 					 crossover=crossover,
# 					 mutation=mutation)

# #Stopping criteria
# stopping_criteria_def = StoppingCriteria(termination_name)
# stopping_criteria = stopping_criteria_def.set_termination(n_gen=n_gen)

# #Obtaining optimal solutions on the initial trained model
# print(f'Performing optimization on the initial trained model using {algorithm_name.upper()}\n')
# optimal_solutions =  do_optimization(TrainedModel_Problem,
# 									 algorithm, stopping_criteria,
# 									 verbose=True, seed=1,
# 									 return_least_infeasible=False)
# print('--------------------------------------------------')
# print('\nOptimal solutions on the initial trained model is obtained!\n')
# print('--------------------------------------------------')

# child_pop = set_individual(X=optimal_solutions.X,
# 						   F=optimal_solutions.F,
# 						   G=optimal_solutions.G,
# 						   CV=optimal_solutions.CV)

# child_pop = set_population_from_array_or_individual(child_pop)

# survivors = Population.merge(parent_pop, child_pop)

# survivors = do_survival(problem, survivors, n_survive=pop_size)

#####################################################################################################

