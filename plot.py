#Plotting the results
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/12/2021
#####################################################################################################
from LoadVars import *
import numpy as np
import matplotlib.pyplot as plt

#Plot
if pf_plot or initial_pop_plot or best_pop_plot or all_pop_plot or all_true_eval_plot:
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
		if len(initial_pop_infeasible)>0:
			plt.plot(initial_pop_infeasible[:,0], initial_pop_infeasible[:,1], 'ro', label='Initial solutions - infeasible')
		if len(initial_pop_feasible)>0:
			plt.plot(initial_pop_feasible[:,0], initial_pop_feasible[:,1], 'bo', label='Initial solutions - feasible')
	
	if best_pop_plot:
		best_pop = np.genfromtxt('OUTPUT/final_pop_FGCV.dat', delimiter=' ')
		best_pop_feasible = np.delete(best_pop, np.where(best_pop[:,-1]>0.0), axis=0)
		best_pop_infeasible = np.delete(best_pop, np.where(best_pop[:,-1]==0.0), axis=0)
		if len(best_pop_infeasible)>0:
			plt.plot(best_pop_infeasible[:,0], best_pop_infeasible[:,1], 'ro', label='Best optimal solutions - infeasible')
		if len(best_pop_feasible)>0:	
			plt.plot(best_pop_feasible[:,0], best_pop_feasible[:,1], 'bo', label='Best optimal solutions - feasible')

	if all_true_eval_plot:
		all_true_eval = np.genfromtxt('OUTPUT/all_true_eval_FGCV.dat', delimiter=' ')
		all_true_eval_feasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]>0.0)[0], axis=0)
		all_true_eval_infeasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]==0.0)[0], axis=0)
		if len(all_true_eval_infeasible)>0:
			plt.plot(all_true_eval_infeasible[:,0],all_true_eval_infeasible[:,1],'ro', markersize=4, label='All solutions - infeasible')
		if len(all_true_eval_feasible)>0:
			plt.plot(all_true_eval_feasible[:,0],all_true_eval_feasible[:,1],'bo', markersize=4, label='All solutions - feasible')
	
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
	plt.legend(loc="center right")
	plt.show()

if igd_plot:
	IGD = np.genfromtxt('OUTPUT/IGD.dat',
		  skip_header=0, skip_footer=0, delimiter=' ')
	plt.plot(IGD[:,1],IGD[:,0])
	plt.title(f'IGD History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("IGD value")
	plt.show()

#####################################################################################################

if all_pop_plot_ga or pf_plot_ga or initial_pop_plot_ga or best_pop_plot_ga or all_true_eval_plot_ga:
	if all_pop_plot_ga:
		all_pop = np.genfromtxt('OUTPUT/PURE_GA/all_pop_FGCV.dat', delimiter=' ')
		all_pop_feasible = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
		all_pop_infeasible = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
		if len(all_pop_infeasible)>0:
			plt.plot(all_pop_infeasible[:,0],all_pop_infeasible[:,1],'ro', markersize=3, label='Infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(all_pop_feasible[:,0],all_pop_feasible[:,1],'bo', markersize=3, label='Feasible')

	if pf_plot_ga:
		pareto_front = np.genfromtxt('OUTPUT/PURE_GA/pareto_front.dat', delimiter=' ')
		plt.plot(pareto_front[:,0],pareto_front[:,1],'k-', label='Pareto front')

	if initial_pop_plot_ga:
		initial_pop = np.genfromtxt('OUTPUT/PURE_GA/initial_pop_FGCV.dat', delimiter=' ')
		initial_pop_feasible = np.delete(initial_pop, np.where(initial_pop[:,-1]>0.0), axis=0)
		initial_pop_infeasible = np.delete(initial_pop, np.where(initial_pop[:,-1]==0.0), axis=0)
		if len(initil_pop_infeasible)>0:
			plt.plot(initial_pop_infeasible[:,0], initial_pop_infeasible[:,1], 'ro', label='Initial solutions - infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(initial_pop_feasible[:,0], initial_pop_feasible[:,1], 'bo', label='Initial solutions - feasible')
	
	if best_pop_plot_ga:
		best_pop = np.genfromtxt('OUTPUT/PURE_GA/final_pop_FGCV.dat', delimiter=' ')
		best_pop_feasible = np.delete(best_pop, np.where(best_pop[:,-1]>00), axis=0)
		best_pop_infeasible = np.delete(best_pop, np.where(best_pop[:,-1]==0.0), axis=0)
		if len(best_pop_infeasible)>0:
			plt.plot(best_pop_infeasible[:,0], best_pop_infeasible[:,1], 'ro', label='Best optimal solutions - infeasible')
		if len(best_pop_feasible)>0:	
			plt.plot(best_pop_feasible[:,0], best_pop_feasible[:,1], 'bo', label='Best optimal solutions - feasible')

	if all_true_eval_plot_ga:
		all_true_eval = np.genfromtxt('OUTPUT/PURE_GA/all_true_eval_FGCV.dat', delimiter=' ')
		all_true_eval_feasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]>00), axis=0)
		all_true_eval_infeasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]==0.0), axis=0)
		if len(all_true_eval_infeasible)>0:
			plt.plot(all_true_eval_infeasible[:,0], all_true_eval_infeasible[:,1], 'ro', markersize=3, label='All solutions - infeasible')
		if len(all_true_eval_feasible)>0:	
			plt.plot(all_true_eval_feasible[:,0], all_true_eval_feasible[:,1], 'bo', markersize=3, label='All solutions - feasible')
	
	plt.title(f'Objective functions space of {problem_name.upper()}')
	plt.xlabel("F1")
	plt.ylabel("F2")
	plt.legend(loc="upper right")
	plt.show()

if hv_plot_ga:

	HV  = np.genfromtxt('OUTPUT/PURE_GA/HV.dat',
		  skip_header=0, skip_footer=0, delimiter=' ')

	HV_pareto = np.genfromtxt('OUTPUT/PURE_GA/HV_pareto.dat')

	plt.plot(HV[:,1],HV[:,0])
	plt.hlines(HV_pareto, 0, len(HV)*pop_size, colors='k', label='Pareto')
	plt.title(f'HV History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("HV value")
	plt.legend(loc="center right")
	plt.show()

if igd_plot_ga:
	IGD = np.genfromtxt('OUTPUT/PURE_GA/IGD.dat',
		  skip_header=0, skip_footer=0, delimiter=' ')
	plt.plot(IGD[:,1],IGD[:,0])
	plt.title(f'IGD History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("IGD value")
	plt.show()

#####################################################################################################

if pf_plot_comp or initial_pop_plot_comp or best_pop_plot_comp or all_pop_plot_comp:
	if all_pop_plot_comp:
		all_pop = np.genfromtxt('OUTPUT/PURE_GA/all_pop_FGCV.dat', delimiter=' ')
		all_pop_feasible = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
		all_pop_infeasible = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
		if len(all_pop_infeasible)>0:
			plt.plot(all_pop_infeasible[:,0],all_pop_infeasible[:,1],'rx', markersize=4, label=f'{algorithm_name.upper()} - Infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(all_pop_feasible[:,0],all_pop_feasible[:,1],'ro', markersize=4, label=f'{algorithm_name.upper()} - Feasible')

		all_pop = np.genfromtxt('OUTPUT/all_pop_FGCV.dat', delimiter=' ')
		all_pop_feasible = np.delete(all_pop, np.where(all_pop[:,-1]>0.0)[0], axis=0)
		all_pop_infeasible = np.delete(all_pop, np.where(all_pop[:,-1]==0.0)[0], axis=0)
		if len(all_pop_infeasible)>0:
			plt.plot(all_pop_infeasible[:,0],all_pop_infeasible[:,1],'bx', markersize=4, label=f'NN+{algorithm_name.upper()} - Infeasible')
		if len(all_pop_feasible)>0:
			plt.plot(all_pop_feasible[:,0],all_pop_feasible[:,1],'bo', markersize=4, label=f'NN+{algorithm_name.upper()} - Feasible')

	if pf_plot_comp:
		pareto_front = np.genfromtxt('OUTPUT/PURE_GA/pareto_front.dat', delimiter=' ')
		plt.plot(pareto_front[:,0],pareto_front[:,1],'k-', label='Pareto front')

	if initial_pop_plot_comp:
		initial_pop = np.genfromtxt('OUTPUT/PURE_GA/initial_pop_FGCV.dat', delimiter=' ')
		initial_pop_feasible = np.delete(initial_pop, np.where(initial_pop[:,-1]>0.0), axis=0)
		initial_pop_infeasible = np.delete(initial_pop, np.where(initial_pop[:,-1]==0.0), axis=0)
		if len(initial_pop_infeasible)>0:
			plt.plot(initial_pop_infeasible[:,0], initial_pop_infeasible[:,1], 'rx', label=f'{algorithm_name.upper()} - Initial solutions - infeasible')
		if len(initial_pop_feasible)>0:
			plt.plot(initial_pop_feasible[:,0], initial_pop_feasible[:,1], 'ro', label=f'{algorithm_name.upper()} - Initial solutions - feasible')

		initial_pop = np.genfromtxt('OUTPUT/initial_pop_FGCV.dat', delimiter=' ')
		initial_pop_feasible = np.delete(initial_pop, np.where(initial_pop[:,-1]>0.0), axis=0)
		initial_pop_infeasible = np.delete(initial_pop, np.where(initial_pop[:,-1]==0.0), axis=0)
		if len(initial_pop_infeasible)>0:
			plt.plot(initial_pop_infeasible[:,0], initial_pop_infeasible[:,1], 'bx', label=f'NN+{algorithm_name.upper()} - Initial solutions - infeasible')
		if len(initial_pop_feasible)>0:
			plt.plot(initial_pop_feasible[:,0], initial_pop_feasible[:,1], 'bo', label=f'NN+{algorithm_name.upper()} - Initial solutions - feasible')
	
	if best_pop_plot_comp:
		best_pop = np.genfromtxt('OUTPUT/PURE_GA/final_pop_FGCV.dat', delimiter=' ')
		best_pop_feasible = np.delete(best_pop, np.where(best_pop[:,-1]>0.0), axis=0)
		best_pop_infeasible = np.delete(best_pop, np.where(best_pop[:,-1]==0.0), axis=0)
		if len(best_pop_infeasible)>0:
			plt.plot(best_pop_infeasible[:,0], best_pop_infeasible[:,1], 'rx', label=f'{algorithm_name.upper()} - Best optimal solutions - infeasible')
		if len(best_pop_feasible)>0:	
			plt.plot(best_pop_feasible[:,0], best_pop_feasible[:,1], 'ro', label=f'{algorithm_name.upper()} - Best optimal solutions - feasible')

		best_pop = np.genfromtxt('OUTPUT/final_pop_FGCV.dat', delimiter=' ')
		best_pop_feasible = np.delete(best_pop, np.where(best_pop[:,-1]>0.0), axis=0)
		best_pop_infeasible = np.delete(best_pop, np.where(best_pop[:,-1]==0.0), axis=0)
		if len(best_pop_infeasible)>0:
			plt.plot(best_pop_infeasible[:,0], best_pop_infeasible[:,1], 'bx', label=f'NN+{algorithm_name.upper()} - Best optimal solutions - infeasible')
		if len(best_pop_feasible)>0:	
			plt.plot(best_pop_feasible[:,0], best_pop_feasible[:,1], 'bo', label=f'NN+{algorithm_name.upper()} - Best optimal solutions - feasible')

	if all_true_eval_plot_comp:
		all_true_eval = np.genfromtxt('OUTPUT/PURE_GA/all_true_eval_FGCV.dat', delimiter=' ')
		all_true_eval_feasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]>00), axis=0)
		all_true_eval_infeasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]==0.0), axis=0)
		if len(all_true_eval_infeasible)>0:
			plt.plot(all_true_eval_infeasible[:,0], all_true_eval_infeasible[:,1], 'rx', markersize=3, label='NSGA2 Infeasible')
		if len(all_true_eval_feasible)>0:	
			plt.plot(all_true_eval_feasible[:,0], all_true_eval_feasible[:,1], 'ro', markersize=3, label='NSGA2 Feasible')

		all_true_eval = np.genfromtxt('OUTPUT/all_true_eval_FGCV.dat', delimiter=' ')
		all_true_eval_feasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]>0.0)[0], axis=0)
		all_true_eval_infeasible = np.delete(all_true_eval, np.where(all_true_eval[:,-1]==0.0)[0], axis=0)
		if len(all_true_eval_infeasible)>0:
			plt.plot(all_true_eval_infeasible[:,0],all_true_eval_infeasible[:,1],'bx', markersize=3, label='NN+NSGA2 Infeasible')
		if len(all_true_eval_feasible)>0:
			plt.plot(all_true_eval_feasible[:,0],all_true_eval_feasible[:,1],'bo', markersize=3, label='NN+NSGA2 Feasible')

	plt.title(f'Objective functions space of {problem_name.upper()}')
	plt.xlabel("F1")
	plt.ylabel("F2")
	plt.legend(loc="upper right")
	plt.show()

if hv_plot_comp:

	HV = np.genfromtxt('OUTPUT/PURE_GA/HV.dat',
		 skip_header=0, skip_footer=0, delimiter=' ')

	HV_pareto = np.genfromtxt('OUTPUT/PURE_GA/HV_pareto.dat')

	plt.plot(HV[:,1],HV[:,0],'r-',label=f'{algorithm_name.upper()}')

	HV = np.genfromtxt('OUTPUT/HV.dat',
		 skip_header=0, skip_footer=0, delimiter=' ')

	HV_pareto = np.genfromtxt('OUTPUT/HV_pareto.dat')

	plt.plot(HV[:,1],HV[:,0],'b-',label=f'NN+{algorithm_name.upper()}')

	plt.hlines(HV_pareto, 0, n_gen_ga*pop_size, colors='k', label='Pareto')


	plt.title(f'HV History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("HV value")
	plt.legend(loc="center right")
	plt.show()

if igd_plot_comp:
	IGD = np.genfromtxt('OUTPUT/PURE_GA/IGD.dat',
		  skip_header=0, skip_footer=0, delimiter=' ')
	plt.plot(IGD[:,1],IGD[:,0],'r-',label=f'{algorithm_name.upper()}')

	IGD = np.genfromtxt('OUTPUT/IGD.dat',
		  skip_header=0, skip_footer=0, delimiter=' ')
	plt.plot(IGD[:,1],IGD[:,0],'b-',label=f'NN+{algorithm_name.upper()}')

	plt.title(f'IGD History of {problem_name.upper()}')
	plt.xlabel("Number of true evaluations")
	plt.ylabel("IGD value")
	plt.legend(loc="center right")
	plt.show()