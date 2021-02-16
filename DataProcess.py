#Data process includes normalization, clustering, cross validation etc
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/16/2021
#####################################################################################################

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

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

def do_gap_statistics(X_nodup, n_var):
	"""
	This function uses gap statistics method to calculate the number of clusters
	Input:
		X_nodup: Training data for the input layer that has no duplication
		n_var: The number of design variables of the problem
	Output:
		N_cluster: the best number of cluster that maximizes gap
	"""
	max_cluster = 30
	trials = 10
	count = np.zeros(max_cluster).astype(np.int32)
	X_rnd = np.random.rand(len(X_nodup), n_var)

	for trial in range(trials):
		gap 	 = np.zeros(max_cluster).astype(np.float32)
		gap_diff = np.zeros(max_cluster).astype(np.float32)
		for cluster in range(max_cluster):
			kmeans       = KMeans(n_clusters=cluster+1).fit(X_nodup)
			kmeans_rnd	 = KMeans(n_clusters=cluster+1).fit(X_rnd)
			gap[cluster] = np.log(kmeans_rnd.inertia_/kmeans.inertia_)

			if cluster==0:
				gap_diff[0] = 0.0
			else:
				gap_diff[cluster] = gap[cluster] - gap[cluster-1]
				if gap_diff[cluster] < 0.0:
					break

		count[np.argmax(gap)] = count[np.argmax(gap)] + 1

	N_cluster = np.argmax(count)+1
	#+1 because cluster in the range(max_cluster) starts from zero
	return N_cluster

def do_KMeans_clustering(N_cluster, X_nodup):
	"""
	This function will use KMeans Clustering method to label training data
	according to its proximity with a cluster
	Input:
		N_cluster: number of cluster estimated by Gap Statistics
		X_nodup: Training data for the input layer that has no duplication
	Output:
		cluster_label: label assigned to every point
		over_coef: this will be used in the oversampling method to increase
				   number of points in the less densed cluster region
	"""

	#Instantiating kmeans object
	kmeans = KMeans(n_clusters=N_cluster).fit(X_nodup)
	cluster_label = kmeans.labels_

	#Calculating the size of cluster (number of data near the cluster centroid)
	cluster_size = np.zeros(N_cluster).astype(np.int32)
	for cluster in range(N_cluster):
		cluster_size[cluster] = len(np.where(cluster_label==cluster)[0])

	over_coef = np.zeros(N_cluster).astype(np.int32)
	for cluster in range(N_cluster):
		over_coef[cluster] = np.copy((max(cluster_size))/cluster_size[cluster])
		if over_coef[cluster] > 10:
			over_coef[cluster] = 10

	return cluster_label, over_coef

def do_oversampling(N_cluster,
					cluster_label,
					X_nodup,
					OUT_nodup,
					over_coef):
	"""
	This function will use oversampling to prevent from overfitting
	Overfitting can happen when training data got stacked in a very densed region
	Oversampling will then be done in the region where cluster size is small
	Input:
		N_cluster: number of cluster estimated by Gap Statistics
		cluster_label: label assigned to every point
		X_nodup: Training data for the input layer that has no duplication
		OUT_nodup: Training data for the output layer that has no duplication
		over_coef: this will be used in the oversampling method to increase
				   number of points in the less densed cluster region
	Output:
		X_over: Training data for the input layer that has been oversampled
		OUT_over: Training data for the output layer that has been oversampled
	"""
	X_over = np.copy(X_nodup)
	OUT_over = np.copy(OUT_nodup)

	for cluster in range(N_cluster):
		idx = np.where(cluster_label!=cluster)[0]
		X_cluster 	= np.delete(X_nodup, idx, axis=0)
		OUT_cluster = np.delete(OUT_nodup, idx, axis=0)

		for counter in range(over_coef[cluster]-1):
			X_over	 = np.vstack((X_over, X_cluster))
			OUT_over = np.vstack((OUT_over, OUT_cluster))

	return X_over, OUT_over

def calc_batchsize(batchrate,
				   train_ratio,
				   X_over):
	"""
	This function calculates the batchsize which is the size of data
	to be processed at once in the training process
	Input:
		train_ratio: the ratio of the data to be used as training set
		batchrate: the percentage from training data to be processed at once
		X_over: Training data for the input layer that has been oversampled
	Output:
		batchsize: The size of data to be processed at once in the training
		N_all: the size of all data after being oversampled
		N_train: the size of training set
		N_valid: the size of validation set
	"""
	N_all = len(X_over)
	N_train = int(N_all*train_ratio)
	N_valid = N_all - N_train

	batchsize = int(batchrate*N_train/100.0)
	if batchsize < 10:
		batchsize = 10
	elif batchsize > 100:
		batchsize = 100

	return batchsize, N_all, N_train, N_valid


def do_cross_validation(N_all, N_train, X_over, OUT_over):
	"""
	This function will separate the data into training and validation set
	This is done to prevent from overfitting
	Validation set is used to guide the learning process of the training set
	Input:
		N_all: the size of all data after being oversampled
		N_train: the size of training set
		train_ratio: the ratio of the data to be used as training set
		X_over: Training data for the input layer that has been oversampled
		OUT_over: Training data for the output layer that has been oversampled
	Output:
		X_train: Training set for the input layer
		OUT_train: Training set for the output layer
		X_valid: Validation set for the input layer
		OUT_valid: Validation set for the output layer
	"""
	#Initializing random permutation
	rand = np.random.permutation(N_all)

	#Separating training set and validation set
	X_train   = X_over[rand[0:N_train]]
	X_valid	  = X_over[rand[N_train:N_all]]
	OUT_train = OUT_over[rand[0:N_train]]
	OUT_valid = OUT_over[rand[N_train:N_all]]

	return X_train, X_valid, OUT_train, OUT_valid