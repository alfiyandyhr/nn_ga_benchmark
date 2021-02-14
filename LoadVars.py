#Loading variables from config.dat
#Outputting variables as python variables loaded in main.py
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/6/2021
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

#Optimization configuration
algorithm_name = config['OPTIMIZATION_ALGORITHM']
selection_operator_name = config['SELECTION_OPERATOR']
crossover_operator_name = config['CROSSOVER_OPERATOR']
prob_c = eval(config['CROSSOVER_PROBABILITY'])
eta_c = eval(config['ETA_CROSSOVER'])
mutation_operator_name = config['MUTATION_OPERATOR']
eta_m = eval(config['ETA_MUTATION'])
termination_name = config['TERMINATION']
n_gen = eval(config['NUMBER_OF_GENERATION'])

#Neural Network configuration
N_Epoch = eval(config['N_EPOCH'])
N_Neuron = eval(config['N_NEURON'])
lr = eval(config['LEARNING_RATE'])
batchrate = eval(config['BATCHRATE'])
number_of_updates = eval(config['NO_OF_UPDATES'])

#Plot config
pf_plot = eval(config['PLOT_PARETO_FRONT'].title())
all_pop_plot = eval(config['PLOT_ALL_POPULATION'].title())
best_pop_plot = eval(config['PLOT_BEST_POPULATION'].title())
initial_pop_plot = eval(config['PLOT_INITIAL_POPULATION'].title())
hv_plot = eval(config['PLOT_HV_HISTORY'].title())
igd_plot = eval(config['PLOT_IGD_HISTORY'].title())

if problem_name == 'osy':
	hv_ref = [-18.8, 83.2]

if problem_name in ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6']:
	hv_ref = [1.1, 1.1]
