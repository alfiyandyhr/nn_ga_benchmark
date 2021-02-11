import numpy as np

def save(f, array, delimiter=' ', newline='\n', header='', footer=''):
	"""
	This function is responsible for any task which includes saving into files
	for example:
		>saving training data
		>saving all_pop.dat
		>saving best_pop.dat
		etc 
	"""

	np.savetxt(f, array,
			   delimiter=delimiter,
			   newline=newline,
			   header=header,
			   footer=footer)