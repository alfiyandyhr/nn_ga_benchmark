import torch
import numpy as np
from pymoo.factory import get_problem
from pymoo.model.population import pop_from_array_or_individual
from pymoo.model.evaluator import Evaluator
import matplotlib.pyplot as plt
from performance import calc_hv

Model = torch.load('DATA/prediction/trained_model.pth')
	# map_location=torch.device('cpu'))

pop_1 = np.genfromtxt('OUTPUT/initial_pop_X.dat')
pop_1_t = torch.from_numpy(pop_1)

pop_1 = pop_from_array_or_individual(pop_1)



problem = get_problem('zdt1')

Evaluator().eval(problem, pop_1)

F = pop_1.get('F')

F_pred = Model(pop_1_t.float()).detach().numpy()

plt.plot(F[:,0],F[:,1],'o',label='True Eval')
plt.plot(F_pred[:,0],F[:,1],'ro',label='Trained NN')
plt.legend(loc='upper right')
plt.show()