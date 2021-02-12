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

	pareto_front = np.genfromtxt('OUTPUT/pareto_front.dat',
				   skip_header=0, skip_footer=0, delimiter=' ')
	best_pop = np.genfromtxt('OUTPUT/final_pop_FGCV.dat',
			   skip_header=0, skip_footer=0, delimiter=' ')
	initial_samples = np.genfromtxt('OUTPUT/initial_pop_FGCV.dat',
				      skip_header=0, skip_footer=0, delimiter=' ')


	if pf_plot:
		plt.plot(pareto_front[:,0],pareto_front[:,1],'k-', label='Pareto front')
	if initial_samples_plot:
		plt.plot(initial_samples[:,0], initial_samples[:,1], 'bo', label='Initial samples')
	if best_pop_plot:
		plt.plot(best_pop[:,0], best_pop[:,1], 'ro', label='Best optimal solutions')
	# plt.plot(optimal_solutions.F[:,0],optimal_solutions.F[:,1],'ro') 
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