#Data process includes normalization, clustering, cross validation etc
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/11/2021
#####################################################################################################

import numpy as np

import matplotlib.pyplot as plt

def normalize(array, v_max, v_min, axis):
	"""
	This function will output a normalized array
	Input:
		a numpy array or tensor (n x m)
			n = batchsize
			m = parameters for every instance
		v_max = maximum value of each parameter/column
		V_min = minimum value of each parameter/column
	Output:
		a normalized numpy array like input
	"""
	n_rows, n_cols = array.shape

	if axis == 0:
		for row in range(n_rows):
			array[row] = (array[row]-v_min)/(v_max-v_min)

	if axis == 1:
		for col in range(n_cols):
			array[:,col] = (array[:,col]-v_min[col])/(v_max[col]-v_min[col])

	return array

def denormalize(array, v_max, v_min, axis):
	"""
	This function will output a denormalized array
	Input:
		a numpy array or tensor (n x m)
			n = batchsize
			m = parameters for every instance
		v_max = maximum value of each parameter/column
		V_min = minimum value of each parameter/column
	Output:
		a normalized numpy array like input
	"""
	n_rows, n_cols = array.shape

	if axis == 0:
		for row in range(n_rows):
			array[row] = (v_max-v_min)*array[row] + v_min

	if axis == 1:
		for col in range(n_cols):
			array[:,col] = (v_max[col]-v_min[col])*array[:,col] + v_min[col]

	return array

def remove_duplicates(X, OUT, n_var):
	"""
	This function will remove any individuals with a distance
	in design space shorter than a specified value
	"""

	X_nodup = np.copy(X)
	OUT_nodup = np.copy(OUT)

	i = 0
	while 1:
		dist = np.sqrt((np.sum(np.power((X_nodup-X_nodup[i,:]), 2.0), axis=1))/n_var)
		dist[i] = 1.0 #make sure the current individual doesnt get deleted
		
		#Location in which the dist is less than 0.001
		idx = np.where(dist<0.001)[0]

		#Deleting the individual at location found above
		X_nodup   = np.delete(X_nodup,   idx, axis=0)
		OUT_nodup = np.delete(OUT_nodup, idx, axis=0)
		
		if len(X_nodup) <= i+1:
			break
		else:
			i = i+1

	return X_nodup, OUT_nodup

