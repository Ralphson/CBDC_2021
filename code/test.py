import numpy as np

a = np.array([[1,-2], [3,4]])
a = np.where(a > 0, a, 100)
print(a)