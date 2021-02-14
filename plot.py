#Plotting the results
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/12/2021
#####################################################################################################
from LoadVars import *
import numpy as np
import matplotlib.pyplot as plt

#Plot
if pf_plot or initial_samples_plot or best_pop_plot:
	if all_pop_plot:
		all_pop = np.genfromtxt('OUTPUT/all_pop_FGCV.dat', delimiter=' ')
		all_pop_feasible = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
		all_pop_infeasible = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
		if len(all_pop_infeasible)>0:
			plt.plot(all_pop_infeasible[:,0],all_pop_infeasible[:,1],'ro', markersize=4, label='Infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(all_pop_feasible[:,0],all_pop_feasible[:,1],'bo', markersize=4, label='Feasible')

	if pf_plot:
		pareto_front = np.genfromtxt('OUTPUT/pareto_front.dat', delimiter=' ')
		plt.plot(pareto_front[:,0],pareto_front[:,1],'k-', label='Pareto front')

	if initial_pop_plot:
		initial_pop = np.genfromtxt('OUTPUT/initial_pop_FGCV.dat', delimiter=' ')
		initial_pop_feasible = np.delete(initial_pop, np.where(initial_pop[:,-1]>0.0), axis=0)
		initial_pop_infeasible = np.delete(initial_pop, np.where(initial_pop[:,-1]==0.0), axis=0)
		if len(initil_pop_infeasible)>0:
			plt.plot(initial_pop_infeasible[:,0], initial_pop_infeasible[:,1], 'ro', label='Initial solutions - infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(initial_pop_feasible[:,0], initial_pop_feasible[:,1], 'bo', label='Initial solutions - feasible')
	
	if best_pop_plot:
		best_pop = np.genfromtxt('OUTPUT/final_pop_FGCV.dat', delimiter=' ')
		best_pop_feasible = np.delete(best_pop, np.where(best_pop[:,-1]>0.0), axis=0)
		best_pop_infeasible = np.delete(best_pop, np.where(best_pop[:,-1]==0.0), axis=0)
		if len(best_pop_infeasible)>0:
			plt.plot(best_pop_infeasible[:,0], best_pop_infeasible[:,1], 'ro', label='Best optimal solutions - infeasible')
		if len(best_pop_feasible)>0:	
			plt.plot(best_pop_feasible[:,0], best_pop_feasible[:,1], 'bo', label='Best optimal solutions - feasible')
	
	plt.title(f'Objective functions space of {problem_name.upper()}')
	plt.xlabel("F1")
	plt.ylabel("F2")
	plt.legend(loc="upper right")
	plt.show()

if hv_plot:

	HV = np.genfromtxt('OUTPUT/HV.dat',
		 skip_header=0, skip_footer=0, delimiter=' ')

	HV_pareto = np.genfromtxt('OUTPUT/HV_pareto.dat')

	plt.plot(HV[:,1],HV[:,0])
	plt.hlines(HV_pareto, 0, len(HV)*pop_size, colors='k', label='Pareto')
	plt.title(f'HV History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("HV value")
	plt.legend(loc="upper right")
	plt.show() 