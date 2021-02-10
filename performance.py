#Performance Indicator Measurement
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/10/2021
#####################################################################################################

def calc_hv(pop, ref):
	"""
	This will calculate Hypervolumes from the input array

	Input: The population at a given generation(numpy)
	
	Output: HV value at a given generation(float)

	"""
	#Sorting the population properly
	pop = pop[pop[:,0].argsort()]

	volume = 0.0

	for indiv in range(len(pop)):
		if indiv == len(pop)-1:
			volume += (ref[0] - pop[indiv,0]) * (ref[1] - pop[indiv,1])
			break
		else:
			volume += (ref[1] - pop[indiv,1]) * (pop[indiv+1,0] - pop[indiv,0])

	return volume
