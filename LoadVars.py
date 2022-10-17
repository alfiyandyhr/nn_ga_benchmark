#Loading variables from config.dat
#Outputting variables as python variables loaded in main.py
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/15/2021
#####################################################################################################

def load_vars():
	"""Loading all varibales and storing them in a dictionary"""
	with open('config.dat') as f:
		content = f.readlines()
		config = {}
		for line in content:
			if line.startswith('%'):
				continue
			item = line.rstrip().split(' = ')
			config[item[0]] = item[1]
	return config

config = load_vars()

"""Assigning variables"""
#Design of Experiment
initial_sampling_method_name = config['SAMPLING_METHOD']
pop_size = eval(config['POPULATION_SIZE'])
problem_name = config['PROBLEM']

#Optimization configuration on the trained NN model
algorithm_name = config['OPTIMIZATION_ALGORITHM']
selection_operator_name = config['SELECTION_OPERATOR']
crossover_operator_name = config['CROSSOVER_OPERATOR']
prob_c = eval(config['CROSSOVER_PROBABILITY'])
eta_c = eval(config['ETA_CROSSOVER'])
mutation_operator_name = config['MUTATION_OPERATOR']
eta_m = eval(config['ETA_MUTATION'])
termination_name = config['TERMINATION']
n_gen = eval(config['NUMBER_OF_GENERATION'])

#Optimization configuration of pure GA (used when use_nn is False)
algorithm_name_ga = config['OPTIMIZATION_ALGORITHM_GA']
selection_operator_name_ga = config['SELECTION_OPERATOR_GA']
crossover_operator_name_ga = config['CROSSOVER_OPERATOR_GA']
prob_c_ga = eval(config['CROSSOVER_PROBABILITY_GA'])
eta_c_ga = eval(config['ETA_CROSSOVER_GA'])
mutation_operator_name_ga = config['MUTATION_OPERATOR_GA']
eta_m_ga = eval(config['ETA_MUTATION_GA'])
termination_name_ga = config['TERMINATION_GA']
n_gen_ga = eval(config['NUMBER_OF_GENERATION_GA'])

#Neural Network configuration
use_nn = eval(config['USE_NN'].title())
N_Epoch = eval(config['N_EPOCH'])
N_Neuron = eval(config['N_NEURON'])
lr = eval(config['LEARNING_RATE'])
train_ratio = eval(config['TRAIN_RATIO'])
batchrate = eval(config['BATCHRATE'])
number_of_updates = eval(config['NO_OF_UPDATES'])

#Plot config for NN surrogate
pf_plot = eval(config['PLOT_PARETO_FRONT'].title())
all_pop_plot = eval(config['PLOT_ALL_POPULATION'].title())
best_pop_plot = eval(config['PLOT_BEST_POPULATION'].title())
initial_pop_plot = eval(config['PLOT_INITIAL_POPULATION'].title())
all_true_eval_plot = eval(config['PLOT_ALL_TRUE_EVAL'].title())
hv_plot = eval(config['PLOT_HV_HISTORY'].title())
igd_plot = eval(config['PLOT_IGD_HISTORY'].title())

#Plot config for pure GA
pf_plot_ga = eval(config['PLOT_PARETO_FRONT_GA'].title())
all_pop_plot_ga = eval(config['PLOT_ALL_POPULATION_GA'].title())
best_pop_plot_ga = eval(config['PLOT_BEST_POPULATION_GA'].title())
initial_pop_plot_ga = eval(config['PLOT_INITIAL_POPULATION_GA'].title())
all_true_eval_plot_ga = eval(config['PLOT_ALL_TRUE_EVAL_GA'].title())
hv_plot_ga = eval(config['PLOT_HV_HISTORY_GA'].title())
igd_plot_ga = eval(config['PLOT_IGD_HISTORY_GA'].title())

#Plot config for comparison
pf_plot_comp = eval(config['PLOT_PARETO_FRONT_COMPARISON'].title())
all_pop_plot_comp = eval(config['PLOT_ALL_POPULATION_COMPARISON'].title())
best_pop_plot_comp = eval(config['PLOT_BEST_POPULATION_COMPARISON'].title())
initial_pop_plot_comp = eval(config['PLOT_INITIAL_POPULATION_COMPARISON'].title())
all_true_eval_plot_comp = eval(config['PLOT_ALL_TRUE_EVAL_COMPARISON'].title())
hv_plot_comp = eval(config['PLOT_HV_HISTORY_COMPARISON'].title())
igd_plot_comp = eval(config['PLOT_IGD_HISTORY_COMPARISON'].title())

if problem_name == 'osy':
	hv_ref = [[-350, 0.0],[-18.8, 83.2]]

if problem_name == 'zdt3':
	hv_ref = [[0.0, -1.0],[1.1, 1.1]]

if problem_name in ['zdt1','zdt2']:
	hv_ref = [[0.0, 0.0],[1.1, 1.1]]