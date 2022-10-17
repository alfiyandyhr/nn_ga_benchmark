#Performance Indicator Measurement
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/17/2021
#####################################################################################################
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

import numpy as np

def calc_hv(pops, ref):
	"""
	This will calculate Hypervolumes from the input array

	Input:
		- pops: The population at a given generation(numpy)
			   It has the first objective in the first column
			   and the second objective in the second column
		- ref: Reference points
	
	Output: HV value at a given generation(double)

	"""
	#Copying pops so that it does not alter the original
	pop = np.copy(pops)

	#Sorting the population properly (from best in obj2)
	pop = pop[pop[:,0].argsort()]

	fronts = fast_non_dominated_sort(pop[:,0:2])

	front_1 = fronts[0]

	pop = pop[front_1]
	
	volume = 0.0

	#Normalization
	for indiv in range(len(pop)):
		pop[indiv,0] = (pop[indiv,0]-ref[0][0])/(ref[1][0]-ref[0][0])
		pop[indiv,1] = (pop[indiv,1]-ref[0][1])/(ref[1][1]-ref[0][1])

	for indiv in range(len(pop)):
		if pop[indiv,-1] == 0.0:
			if indiv == len(pop)-1:
				volume += (1.0 - pop[indiv,0]) * (1.0 - pop[indiv,1])
				if volume < 0.0:
					volume = 0.0
				break
			else:
				volume += (1.0 - pop[indiv,1]) * (pop[indiv+1,0] - pop[indiv,0])
				if volume < 0.0:
					volume = 0.0
		
	return volume

def calc_igd(pops, ref, pfs):
	"""
	This will calculate the inverted generational distance

	Input:
		pops: The population at a given generation(numpy)
		ref: Reference points
		pfs: Pareto front of the problem
	
	Output: IGD value at a given generation(double)

	"""	
	#Copying pops and pfs so that it does not alter the original
	pop = np.copy(pops)
	pf = np.copy(pfs)

	#Normalization
	for indiv in range(len(pop)):
		pop[indiv,0] = (pop[indiv,0]-ref[0][0])/(ref[1][0]-ref[0][0])
		pop[indiv,1] = (pop[indiv,1]-ref[0][1])/(ref[1][1]-ref[0][1])

	for indiv_pf in range(len(pf)):
		pf[indiv_pf,0] = (pf[indiv_pf,0]-ref[0][0])/(ref[1][0]-ref[0][0])
		pf[indiv_pf,1] = (pf[indiv_pf,1]-ref[0][1])/(ref[1][1]-ref[0][1])

	sum_igd = 0.0
	for indiv_pf in range(len(pf)):
		min_igd = 1.0E5
		for indiv in range(len(pop)):
			if pop[indiv,-1] == 0.0:
				igd = np.sqrt(np.power((pf[indiv_pf,0]-pop[indiv,0]),2)+np.power((pf[indiv_pf,1]-pop[indiv,1]),2)) 
				if igd < min_igd:
					min_igd = igd
		sum_igd += min_igd

	return sum_igd

def monotonous_IGD(IGD):
	"""
	This function will make the IGD values
	monotonously decreasing for easier analysis
	"""
	for i in range(len(IGD)-1):
		if IGD[i+1,0] > IGD[i,0]:
			IGD[i+1,0] = IGD[i,0]
	return IGD