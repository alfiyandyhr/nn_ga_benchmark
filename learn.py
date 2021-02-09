#NeuralNet routines, including training and predicting
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/6/2021
#####################################################################################################
import torch
import numpy as np

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

def train(model, N_Epoch, init, device='cpu'):
	"""Training routines"""
	#Loading training data
	if init:
		x = np.genfromtxt('DATA/training/X_init.dat',
			skip_header=1, skip_footer=0, delimiter=' ')
		y = np.genfromtxt('DATA/training/OUT_init.dat',
			skip_header=1, skip_footer=0, delimiter=' ')
	else:
		x = np.genfromtxt('DATA/training/X.dat',
			skip_header=1, skip_footer=0, delimiter=' ')
		y = np.genfromtxt('DATA/training/OUT.dat',
			skip_header=1, skip_footer=0, delimiter=' ')

	#Converting training data to pytorch tensors
	x_t = torch.from_numpy(x)
	y_t = torch.from_numpy(y)

	x_t = x_t.to(device)
	y_t = y_t.to(device)

	print(x_t.device)

	#Defining loss functions and parameter optimizers
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)

	#Training
	for t in range(N_Epoch):
		y_pred = model(x_t.float())
		loss = loss_fn(y_pred, y_t.float())
		if t % 100 == 99:
		    print(f'N_Epoch = {t}', f'Loss = {loss.item()}')
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	#Returning a trained model
	return loss_fn

#Implement cuda computation if exists
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

device = torch.device('cpu')

N_vars, N_neurons, N_objs, N_Epoch = 30, 32, 2, 10000
Model = NeuralNet(D_in=N_vars,
				  H=N_neurons,
				  D=N_neurons,
				  D_out=N_objs).to(device)
TrainedModel = train(Model, N_Epoch, init=True, device=device)

print(TrainedModel.device)
print(torch.cuda.memory_reserved())