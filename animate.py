import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure()
ax = plt.axes(xlim=(-0.1,1.1),ylim=(-0.1,7.1))
ln1, = plt.plot([], [], 'bo', markersize=2, label='NN+GA')
ln2, = plt.plot([], [], 'ro', markersize=2, label='NSGA2')
ln3, = plt.plot([], [], 'k-', label='Pareto Front')
x, y_nn, y_nsga2 = [], [], []

all_pop = np.genfromtxt('OUTPUT/PURE_GA/all_pop_FGCV.dat', delimiter=' ')
all_pop_feasible_ga = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
all_pop_infeasible_ga = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
all_pop = np.genfromtxt('OUTPUT/all_pop_FGCV.dat', delimiter=' ')
all_pop_feasible_nn = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
all_pop_infeasible_nn = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
pareto_front = np.genfromtxt('OUTPUT/PURE_GA/pareto_front.dat', delimiter=' ')

def init():
	ln1.set_data([],[])
	ln2.set_data([],[])
	ln3.set_data(pareto_front[:,0],pareto_front[:,1])


def update(i):
	x_nsga2 = all_pop_feasible_ga[0:100*i,0]
	y_nsga2 = all_pop_feasible_ga[0:100*i,1]
	ln2.set_data(x_nsga2, y_nsga2)
	x_nn = all_pop_feasible_nn[0:100*i,0]
	y_nn = all_pop_feasible_nn[0:100*i,1]
	ln1.set_data(x_nn, y_nn)
	
ani = FuncAnimation(fig, update, init_func=init)

writer = PillowWriter(fps=25)
ani.save("PLOT/All_pop_NSGA2_vs_NN+GA.gif", writer=writer)

plt.show()
