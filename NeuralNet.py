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

def train(model, N_Epoch, lr, init):
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

	#Defining loss functions and parameter optimizers
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)

	#Training
	for t in range(N_Epoch):
		y_pred = model(x_t.float())
		loss = loss_fn(y_pred, y_t.float())
		if t % 200 == 99:
		    print(f'N_Epoch = {t}', f'Loss = {loss.item()}')
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	y_pred_np = y_pred.detach().numpy()

	np.savetxt('DATA/prediction/objs_constrs_init.dat',
		y_pred_np, delimiter=' ', newline='\n',
		header='Row = sample point, column = objective functions and constraint functions',
		footer="")
	
	#Returning a trained model
	return model