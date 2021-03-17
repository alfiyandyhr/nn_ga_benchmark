import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure()
# ax = plt.axes(xlim=(-1600,50),ylim=(-10,300))
ax = plt.axes(xlim=(-0.1,1.1),ylim=(-0.5,7))
ln1, = plt.plot([], [], 'bo', markersize=2, label='NN+GA')
ln2, = plt.plot([], [], 'ro', markersize=2, label='NSGA2')
ln3, = plt.plot([], [], 'k-', label='Pareto Front')
ln4, = plt.plot([], [], 'bx', markersize=2, label='NN+GA')
ln5, = plt.plot([], [], 'rx', markersize=2, label='NSGA2')
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
	ln4.set_data([],[])
	ln5.set_data([],[])


def update(i):
	x_nsga2_feas = all_pop_feasible_ga[0:100*i,0]
	y_nsga2_feas = all_pop_feasible_ga[0:100*i,1]
	ln2.set_data(x_nsga2_feas, y_nsga2_feas)
	x_nn_feas = all_pop_feasible_nn[0:100*i,0]
	y_nn_feas = all_pop_feasible_nn[0:100*i,1]
	ln1.set_data(x_nn_feas, y_nn_feas)
	x_nsga2_infeas = all_pop_infeasible_ga[0:100*i,0]
	y_nsga2_infeas = all_pop_infeasible_ga[0:100*i,1]
	ln5.set_data(x_nsga2_infeas, y_nsga2_infeas)
	x_nn_infeas = all_pop_infeasible_nn[0:100*i,0]
	y_nn_infeas = all_pop_infeasible_nn[0:100*i,1]
	ln4.set_data(x_nn_infeas, y_nn_infeas)
	
ani = FuncAnimation(fig, update, init_func=init)

writer = PillowWriter(fps=25)
ani.save("PLOT/All_pop_NSGA2_vs_NN+GA.gif", writer=writer)

plt.show()
