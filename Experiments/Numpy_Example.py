print('hello world')
import tensorflow
from matplotlib import pyplot as plt
import numpy as np

a = np.random.random((100,100))
a[50,50] = 100
a[28,21] = 100

plt.imshow(a)
plt.show()