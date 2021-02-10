from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator

from performance import *

import numpy as np
import matplotlib.pyplot as plt

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=False)

n_evals = []
F =[]
cv = []

for algorithm in res.history:
	n_evals.append(algorithm.evaluator.n_eval)

	opt = algorithm.opt

	cv.append(opt.get('CV').min())

	feas = np.where(opt.get('feasible'))[0]
	_F = opt.get('F')[feas]
	F.append(_F)


HV = []

for gen in range(len(F)):
	HV.append(calc_hv(F[gen], [1.1,1.1]))

HV_pareto = calc_hv(problem.pareto_front(),[1.1,1.1])

plt.plot(n_evals, HV)
plt.hlines(HV_pareto,0,n_evals[len(n_evals)-1],colors='k')
plt.show()

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()