#NeuralNet routines, including training and predicting
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/16/2021
#####################################################################################################
import torch
import numpy as np
from DataProcess import normalize, denormalize, remove_duplicates
from DataProcess import do_gap_statistics
from DataProcess import do_KMeans_clustering, do_oversampling
from DataProcess import calc_batchsize, do_cross_validation


class NeuralNet(torch.nn.Module):
	"""A neural net architecture"""
	def __init__(self, D_in, H, D, D_out):
		"""Inheritance from torch.nn.Module"""
		super(NeuralNet, self).__init__()
		self.input_layer = torch.nn.Linear(D_in, H)
		self.hidden_layer1 = torch.nn.Linear(H, H)
		self.hidden_layer2 = torch.nn.Linear(H, D)
		self.output_layer = torch.nn.Linear(D, D_out)

		#NeuralNet config
		self.D_in = D_in
		self.H = H
		self.D = D
		self.D_out = D_out

	def forward(self, x):
		"""Forward propagation"""
		y_pred = self.output_layer(self.PHI(x))
		return y_pred

	def PHI(self, x):
		"""Propagation in between"""
		h_relu = self.input_layer(x).tanh()
		for i in range(2):
		    h_relu = self.hidden_layer1(h_relu).tanh()
		phi = self.hidden_layer2(h_relu)
		return phi

def train(problem,
		  model,
		  N_Epoch,
		  lr,
		  train_ratio,
		  batchrate,
		  device):
	"""
	Training routines
	"""

	#Loading training data
	X = np.genfromtxt('DATA/training/X.dat',
		skip_header=0, skip_footer=0, delimiter=' ')
	OUT = np.genfromtxt('DATA/training/OUT.dat',
		skip_header=0, skip_footer=0, delimiter=' ')

	print('Processing the data... please wait :)\n')

	X   = torch.from_numpy(X).float().to(device)
	OUT = torch.from_numpy(OUT).float().to(device)

	#Normalization of input and output
	OUT_max = torch.amax(OUT, axis=0)
	OUT_min = torch.amin(OUT, axis=0)

	X = normalize(X, problem.xu, problem.xl, axis=0)
	OUT = normalize(OUT, OUT_max, OUT_min, axis=1)

	"""
	Remove duplicates from training data
	Duplicates of training data might add weights to them
	"""
	X, OUT = remove_duplicates(X, OUT, problem.n_var)

	"""
	Gap statistics
	"""
	N_cluster = do_gap_statistics(X, problem.n_var)

	"""
	Clustering
	"""
	cluster_label, over_coef = do_KMeans_clustering(N_cluster, X)

	"""
	Over sampling
	"""
	X, OUT = do_oversampling(N_cluster, cluster_label,
							 X, OUT, over_coef)

	"""
	Setting for batch processing
	"""
	batchsize, N_all, N_train, N_valid = calc_batchsize(batchrate, train_ratio, X)

	"""
	Cross Validation
	"""
	X_train, X_valid, OUT_train, OUT_valid = do_cross_validation(N_all, N_train,
																 X, OUT)

	#Defining loss functions and parameter optimizers
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)

	train_lost = torch.zeros(N_Epoch).to(device)
	valid_lost  = torch.zeros(N_Epoch).to(device)
	valid_loss_min = torch.tensor(float('inf'))

	#Training
	for epoch in range(N_Epoch):
		#Monitor losses
		train_loss = 0.0
		valid_loss = 0.0

		perm = torch.randperm(N_train)

		###################
		# Train the model #
		###################
		model.train().to(device)
		for i in range(0, N_train, batchsize):
			optimizer.zero_grad()
			OUT_pred_train = model(X_train[perm[i:i+batchsize]].float())
			loss = loss_fn(OUT_pred_train, OUT_train[perm[i:i+batchsize]].float())
			loss.backward()
			torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
			optimizer.step()
			train_loss += loss.item()*batchsize

		######################
		# Validate the model #
		######################
		model.eval().to(device)
		OUT_pred_valid = model(X_valid[0:N_valid].float())
		loss = loss_fn(OUT_pred_valid, OUT_valid[0:N_valid].float())
		valid_loss = loss.item()

		#Average loss over an epoch
		train_lost[epoch] = train_loss/N_train
		valid_lost[epoch] = valid_loss

		if epoch % 50 == 0:
		    print(f'N_Epoch = {epoch}, Train Loss = {train_lost[epoch]}, Valid Loss = {valid_loss}')

		#Save the model when validation loss has decreased
		if valid_loss <= valid_loss_min:
			valid_loss_min=valid_loss
			torch.save(model, 'DATA/prediction/trained_model.pth')

		if epoch>=50 and torch.mean(valid_lost[epoch-25:epoch])-torch.mean(valid_lost[epoch-50:epoch-25])>0:
			break

def calculate(X, problem, device):
	"""
	This function is used as the approximate function evaluation used in GA
	(called in ga.py under class TrainedModelProblem)
	
	Input: denormalized array of design variables

	This function will normalize the input and denormalize the output

	Output: denormalized array of objectives and constraints

	"""
	#Converting to tensor
	"""
	no need to convert problem.xu and problem.xl into tensors
	because they are automatically converted by pytorch when
	operations between tensors and numpy arrays happen
	"""
	X = torch.from_numpy(X).to(device)

	#Normalization of input
	X = normalize(X, problem.xu, problem.xl, axis=0)

	#Loading the model
	model = torch.load('DATA/prediction/trained_model.pth').to(device)

	#Trained model produces output
	OUT = model(X.float())

	#Denormalization of output
	out = np.genfromtxt('DATA/training/OUT.dat',
	skip_header=0, skip_footer=0, delimiter=' ')

	out = torch.from_numpy(out).to(device)

	OUT_max = torch.amax(out, axis=0)
	OUT_min = torch.amin(out, axis=0)

	OUT = denormalize(OUT, OUT_max, OUT_min, axis=1)

	return OUT.detach().numpy()



