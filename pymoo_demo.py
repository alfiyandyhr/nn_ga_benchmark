from pymoo.factory import get_problem, get_sampling
from pymoo.model.individual import Individual
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize

problem = get_problem('osy')
sampling = get_sampling('real_lhs')
pop_size = 100


X_init = sampling.do(problem, n_samples=pop_size, pop=None)
F_init = problem.evaluate(X_init, return_values_of=['F'])
G_init = problem.evaluate(X_init, return_values_of=['G'])
CV_init = problem.evaluate(X_init, return_values_of=['CV'])

parent_pop = Individual(X=X_init,
						F=F_init,
						G=G_init,
						CV=CV_init)

parent_pop = pop_from_array_or_individual(parent_pop)

algorithm = NSGA2(pop_size=pop_size)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=True)

child_pop = Individual(X=res.X,
					  F=res.F,
					  G=res.G,
					  CV=res.CV)

merged_pop = Population.merge(parent_pop, child_pop)

Survival = RankAndCrowdingSurvival()

parent_pop = Survival.do(problem, merged_pop, n_survive=pop_size)


