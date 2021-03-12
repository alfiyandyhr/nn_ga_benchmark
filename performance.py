#Performance Indicator Measurement
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/12/2021
#####################################################################################################
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

import numpy as np

def calc_hv(pop, ref):
	"""
	This will calculate Hypervolumes from the input array

	Input:
		- pop: The population at a given generation(numpy)
			   It has the first objective in the first column
			   and the second objective in the second column
		- ref: Reference point
	
	Output: HV value at a given generation(double)

	"""
	#Sorting the population properly (from best in obj1)
	pop = pop[pop[:,0].argsort()]

	fronts = fast_non_dominated_sort(pop[:,0:2])

	front_1 = fronts[0]

	volume = 0.0

	for indiv in front_1:
		if pop[indiv,-1] == 0.0:
			if indiv == front_1[-1]:
				volume += (ref[0] - pop[indiv,0]) * (ref[1] - pop[indiv,1])
				break
			else:
				volume += (ref[1] - pop[indiv,1]) * (pop[indiv+1,0] - pop[indiv,0])

	if volume < 0.0:
		volume = 0.0
		
	return volume

def calc_igd(pop, pf):
	"""
	This will calculate the inverted generational distance

	Input:
		pop: The population at a given generation(numpy)
		pf: Pareto front of the problem
	
	Output: IGD value at a given generation(double)

	"""
	fronts = fast_non_dominated_sort(pop[:,0:2])

	front_1 = fronts[0]

	sum_igd = 0.0

	for indiv_pf in range(len(pf)):
		min_igd = 1.0E5
		for indiv in front_1:
			if pop[indiv,-1] == 0.0:
				igd = np.sqrt(np.power((pf[indiv_pf,0]-pop[indiv,0]),2)+np.power((pf[indiv_pf,0]-pop[indiv,1]),2)) 
				if igd < min_igd:
					min_igd = igd
		sum_igd += min_igd

	return sum_igd