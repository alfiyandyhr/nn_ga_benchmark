This is the Readme file for NN+GA code for the benchmark problems.

Author: Alfiyandy Hariansyah
Institution: Institute of Fluid Science, Tohoku University                   
Date: 10/18/2022    
Deep Learning Framework: PyTorch        
Optimization: Genetic Algorithms (Framework: pymoo)   
Language: Python
--------------------------------------------------------------------------

Requirements
---------------------------------------------------------------------------
This code depends on the following python libraries:
> numpy (https://numpy.org/install/)
> scipy (https://scipy.org/install/)
> matplotlib (https://matplotlib.org/stable/users/installing.html)
> pymoo (https://pymoo.org/installation.html)
> pytorch (https://pytorch.org/get-started/locally/)

How to compile and run the program
---------------------------------------------------------------------------
Edit config.dat file to match the user needs
The file containts python variables used during the routines

All the routines are executed in the main.py file
> to execute: 'python main.py'

Plot:
> to execute: 'python plot.py'

Animation:
> to execute: 'python animate.py'
---------------------------------------------------------------------------

About the output files
---------------------------------------------------------------------------
OUTPUT/initial_pop_X.dat    : the design variables of the initial population
OUTPUT/initial_pop_FGCV.dat : the objectives, constraints, and CVs of the initial population
OUTPUT/all_pop_X.dat        : the design variables of all populations
OUTPUT/all_pop_FGCV.dat     : the objectives, constraints, and CVs of all populations
OUTPUT/all_true_eval.dat    : the objectives, constraints, and CVs of all true evaluations
OUTPUT/final_pop_X.dat      : the design variables of the final population (best)
OUTPUT/final_pop_FGCV.dat   : the objectives, constraints, and CVs of the final population (best)
OUTPUT/HV.dat               : the hypervolume values history
OUTPUT/IGD.dat              : the inverted generational distance values history

OUTPUT/PURE_GA/initial_pop_X.dat    : the design variables of the initial population
OUTPUT/PURE_GA/initial_pop_FGCV.dat : the objectives, constraints, and CVs of the initial population
OUTPUT/PURE_GA/all_pop_X.dat        : the design variables of all populations
OUTPUT/PURE_GA/all_pop_FGCV.dat     : the objectives, constraints, and CVs of all populations
OUTPUT/PURE_GA/all_true_eval.dat    : the objectives, constraints, and CVs of all true evaluations
OUTPUT/PURE_GA/final_pop_X.dat      : the design variables of the final population (best)
OUTPUT/PURE_GA/final_pop_FGCV.dat   : the objectives, constraints, and CVs of the final population (best)
OUTPUT/PURE_GA/HV.dat               : the hypervolume values history
OUTPUT/PURE_GA/IGD.dat              : the inverted generational distance values history

DATA/training/X.dat         : the design variables input (fed to the input layer of NN)
DATA/training/OUT.dat       : the functions to be modeled (fed to the output layer of NN)

DATA/prediction/trained_model.pth : this contains the neural net architecture with its weights
---------------------------------------------------------------------------

About the input parameters
---------------------------------------------------------------------------
n_var: number of design variables
n_obj: number of objectives
n_constr: number of constraints
pop_size: the population size
n_gen: number of generations for the GA on the trained model
prob_c: crossover probability
eta_c: eta crossover
eta_m: eta mutation
N_Epoch = number of epochs allowed in the training
N_Neuron = number of neurons in the hidden layers
lr = the learning rate of the training
train_ratio = the ration of the training data to all data
batchrate = the percentage of the batch to train the data
number_of_updates = number to update the model (not including the initial training)
---------------------------------------------------------------------------

Defining the optimization problem
---------------------------------------------------------------------------
1. Edit the file 'config.dat', line 22
2. Available problems > https://pymoo.org/problems/index.html
---------------------------------------------------------------------------

About the files
---------------------------------------------------------------------------
In the main directory:

config.dat: the user interface of input variables
main.py: the main executable file (>python main.py)
plot.py: the plot executable file (>python plot.py)
animate.py: the animation plot of all population dynamics (>python animate.py)

In the module nnga:

LoadVars.py: the file responsible for reading the 'config.dat'
ga.py: the file responsible for optimization routines (pymoo framework)
NeuralNet.py: the file responsible for the NN models (architectures, training, calculating, etc)
DataProcess.py: the file responsible for treating the data before being fed to the neural net
kmeans.py: the file that conducts kmeans clustering used in DataProcess.py
SaveOutput.py: the file responsible for saving the output files
eval.py: the file responsible for the true objective and constraint evaluations
performance.py: the file that assesses the performance using HV and IGD
test_model.py: the playground to test the trained NN model

---------------------------------------------------------------------------

Routines (STEP BY STEP)
---------------------------------------------------------------------------
0. Edit the config file ('config.dat')
      > if you want to use NN+GA, set line 69 to ('USE_NN = TRUE')
      > if you want to use PURE_GA, set line 60 to ('USE_NN = FALSE')
1. Edit the class 'NeuralNet' in the file 'nnga/NeuralNet.py' to build
   the neural net structure depending on the problem (trial-and-error)
   by default: 3 hidden layers with two layers sharing the same weights
2. Execute 'python main.py'
3. Execute 'python plot.py'
4. Execute 'python animate.py'
5. See the output files in the OUTPUT folder
6. See the GIF animation in the PLOT folder
---------------------------------------------------------------------------

Acknowledgements
> Kato-san and Shimoyama-sensei at IFS, TU
> Julian and Deb at MSU Coin Lab

Please feel free to send questions/comments/doubts/suggestions/bugs
etc. to muhammad.alfiyandy.hariansyah.s8@dc.tohoku.ac.jp

---------------------------------------------------------------------------