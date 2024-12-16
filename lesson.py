import numpy as np
import math
a = np.array([1,2,3])
b = np.arange(50,60)
c = np.concatenate((a,b))
num_expand_x = math.ceil(
    64 * 70 / 1000)
print(num_expand_x)