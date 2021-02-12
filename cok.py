import numpy as np

a = [1, 2, 3]
a = np.array([a]).T
b = [3, 4, 5]
b = np.array([b]).T

c = np.concatenate((a,b), axis = 1)

print(c)