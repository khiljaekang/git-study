import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    return list(map(lambda x : x if x > 0 else 1*(np.exp(x)-1), x))

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()    