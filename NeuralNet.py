#NeuralNet routines, including training and predicting
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/6/2021
#####################################################################################################
import torch
import numpy as np
from sklearn.cluster import KMeans

class NeuralNet(torch.nn.Module):
	"""A neural net instance"""
	def __init__(self, D_in, H, D, D_out):
		"""Inheritance from torch.nn.Module"""
		super(NeuralNet, self).__init__()
		self.inputlayer = torch.nn.Linear(D_in, H)
		self.middle = torch.nn.Linear(H, H)
		self.lasthiddenlayer = torch.nn.Linear(H, D)
		self.outputlayer = torch.nn.Linear(D, D_out)

		#NeuralNet config
		self.D_in = D_in
		self.H = H
		self.D = D
		self.D_out = D_out

	def forward(self, x):
		"""Forward propagation"""
		y_pred = self.outputlayer(self.PHI(x))
		return y_pred

	def PHI(self, x):
		"""Propagation in between"""
		h_relu = self.inputlayer(x).tanh()
		for i in range(2):
		    h_relu = self.middle(h_relu).tanh()
		phi = self.lasthiddenlayer(h_relu)
		return phi

def train(problem, model, N_Epoch, lr, batchrate):
	"""Training routines"""
	#Loading training data
	X = np.genfromtxt('DATA/training/X.dat',
		skip_header=0, skip_footer=0, delimiter=' ')
	OUT = np.genfromtxt('DATA/training/OUT.dat',
		skip_header=0, skip_footer=0, delimiter=' ')

	#Normalization of input and output
	n_rows_X, n_cols_X = X.shape
	n_rows_OUT, n_cols_OUT = OUT.shape

	for row in range(n_rows_X):
		X[row] = (X[row]-problem.xl)/(problem.xu-problem.xl)

	for row in range(n_rows_OUT):
		for col in range(n_cols_OUT):
			OUT_max = np.max(OUT[:,col])
			OUT_min = np.min(OUT[:,col])
			OUT[row,col] = (OUT[row,col]-OUT_min)/(OUT_max-OUT_min)

	"""
	Remove duplicates from training data
	Duplicates of training data might add weights to them
	"""
	X_nodup   = np.copy(X)
	OUT_nodup = np.copy(OUT)
	
	i = 0
	while 1:
		dist = np.sqrt((np.sum(np.power((X_nodup-X_nodup[i,:]), 2.0), axis=1))/problem.n_var)
		dist[i] = 1.0
		
		idx = np.where(dist<0.001)[0]
		X_nodup   = np.delete(X_nodup,   idx, 0)
		OUT_nodup = np.delete(OUT_nodup, idx, 0)
		
		if len(X_nodup) <= i+1:
			break
		else:
			i = i+1

	"""
	Gap statistics
	"""
	max_cluster = 30
	trial = 10
	count = np.zeros(max_cluster).astype(np.int32)
	X_rnd = np.random.rand(len(X_nodup), problem.n_var)
	
	for i in range(trial):
		gap = np.zeros(max_cluster).astype(np.float32)
		gap_diff = np.zeros(max_cluster).astype(np.float32)
		for j in range(max_cluster):
			kmeans     = KMeans(n_clusters=j+1).fit(X_nodup)
			kmeans_rnd = KMeans(n_clusters=j+1).fit(X_rnd)
			gap[j] = np.log(kmeans_rnd.inertia_/kmeans.inertia_)
			
			if j==0:
				gap_diff[0] = 0.0
			else:
				gap_diff[j] = gap[j] - gap[j-1]
				if gap_diff[j] < 0:
					break
		
		count[np.argmax(gap)] = count[np.argmax(gap)]+1
	
	N_cluster = np.argmax(count)+1

	"""
	Clustering
	"""
	kmeans = KMeans(n_clusters=N_cluster).fit(X_nodup)
	cluster_label = kmeans.labels_
	
	cluster_size = np.zeros(N_cluster).astype(np.int32)
	for i in range(N_cluster):
		cluster_size[i] = len(np.where(cluster_label==i)[0])
	
	over_coef = np.zeros(N_cluster).astype(np.int32)
	for i in range(N_cluster):
		over_coef[i] = np.copy((max(cluster_size))/cluster_size[i])
		if over_coef[i] > 10:
			over_coef[i] = 10

	"""
	Over sampling
	"""
	X_over    = np.copy(X_nodup)
	OUT_over  = np.copy(OUT_nodup)
	
	# for i in range(N_cluster):
	# 	idx = np.where(cluster_label!=i)[0]
	# 	X_cluster   = np.delete(X_nodup,   idx, 0)
	# 	OUT_cluster = np.delete(OUT_nodup, idx, 0)
		
	# 	for j in range(over_coef[i]-1):
	# 		X_over   = np.vstack((X_over,   X_cluster))
	# 		X_over = np.vstack((OUT_over, OUT_cluster))


	"""
	Setting
	"""
	train_ratio = 0.8
	N_all   = len(X_over)
	N_train = int(N_all*train_ratio)
	N_test  = N_all-N_train
	
	batchsize = int(batchrate*N_train/100.0)
	if batchsize < 10:
		batchsize = 10
	if batchsize > 100:
		batchsize = 100

	"""
	Cross Validation: Separate training and test datas
	"""
	rand = np.random.permutation(N_all)
	X_train   = X_over[rand[0:N_train]]
	X_test    = X_over[rand[N_train:N_all]]
	OUT_train = OUT_over[rand[0:N_train]]
	OUT_test  = OUT_over[rand[N_train:N_all]]

	#Converting training data to pytorch tensors
	X_train = torch.from_numpy(X_train)
	X_test = torch.from_numpy(X_test)
	OUT_train = torch.from_numpy(OUT_train)
	OUT_test = torch.from_numpy(OUT_test)

	#Defining loss functions and parameter optimizers
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)

	train_lost = np.zeros(N_Epoch)
	test_lost  = np.zeros(N_Epoch)
	min_test_lost = 1.0e10

	#Training
	for epoch in range(N_Epoch):

		sum_train_loss = 0
		perm = np.random.permutation(N_train)

		for i in range(0, N_train, batchsize):
			optimizer.zero_grad()
			OUT_pred_train = model(X_train[perm[i:i+batchsize]].float())
			train_loss = loss_fn(OUT_pred_train, OUT_train[perm[i:i+batchsize]].float())
			# print(f'i = {i}', f'Loss = {train_loss.item()}')
			train_loss.backward()
			optimizer.step()
			sum_train_loss += train_loss.data*batchsize
		# if epoch % 10 == 0:
		    # print(f'N_Epoch = {epoch}', f'Loss = {train_loss.item()}')
		train_lost[epoch] = sum_train_loss/N_train


		model = model.eval()
		OUT_pred_test = model(X_test[0:N_test].float())
		test_loss = loss_fn(OUT_pred_test, OUT_test[0:N_test].float())

		test_lost[epoch] = test_loss.data

		if min_test_lost-test_lost[epoch]>0:
			min_test_lost=test_lost[epoch]
			TrainedModel = model

		if epoch>=50 and np.average(test_lost[epoch-25:epoch])-np.average(test_lost[epoch-50:epoch-25])>0:
			break

		# OUT_pred = model(X_t.float())
		# loss = loss_fn(OUT_pred, OUT_t.float())
		# if epoch % 200 == 99:
		#     print(f'N_Epoch = {t}', f'Loss = {loss.item()}')
		# optimizer.zero_grad()
		# loss.backward()
		# optimizer.step()
	
	#Returning a trained model
	return TrainedModel

def calculate(X, problem, model):
	"""
	This function is used as the approximate function evaluation used in GA
	(called in ga.py under class TrainedModelProblem)
	
	Input: denormalized array of design variables

	This function will normalize the input and denormalize the output

	Output: denormalized array of objectives and constraints

	"""
	#Converting to tensor
	X_t = torch.from_numpy(X)
	X_t_max = torch.from_numpy(problem.xu)
	X_t_min = torch.from_numpy(problem.xl)

	#Normalization of input
	n_rows, n_cols = X_t.shape
	
	for row in range(n_rows):
		X_t[row] = (X_t[row]-X_t_min)/(X_t_max-X_t_min)

	#Trained model produces output
	out_t = model(X_t.float())

	#Denormalization of output
	n_rows, n_cols = out_t.shape

	for row in range(n_rows):
		for col in range(n_cols):
			out_t_max = torch.max(out_t[:,col])
			out_t_min = torch.min(out_t[:,col])
			out_t[row,col] = (out_t_max-out_t_min) * out_t[row,col] + out_t_min

	#Converting to numpy from tensor
	out = out_t.detach().numpy()

	return out



