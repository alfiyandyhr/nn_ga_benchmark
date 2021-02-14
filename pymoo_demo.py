from pymoo.factory import get_problem, get_sampling
from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.optimize import minimize

from SaveOutput import save

problem = get_problem('osy')
problem2 = get_problem('zdt1')
sampling = get_sampling('real_lhs')
pop_size = 100

parent_pop = sampling.do(problem, n_samples=pop_size)

Evaluator().eval(problem, parent_pop)

algorithm = NSGA2(pop_size=pop_size)

res = minimize(problem,
			   algorithm,
			   ('n_gen', 2),
			   seed=1,
			   save_history=True,
			   verbose=True)

child_pop = pop_from_array_or_individual(res.X)

print(child_pop.get('G'))

Evaluator().eval(problem, child_pop)

print(child_pop.get('G'))

# merged_pop = Population.merge(parent_pop, child_pop)

# Survival = RankAndCrowdingSurvival()

# parent_pop = Survival.do(problem, merged_pop, n_survive=pop_size)

# print(parent_pop.get('X').shape)

