import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 1.67*(np.exp(x)-1)
        y_list.append(y)
    return y_list

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()