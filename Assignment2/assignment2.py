''' Import Python Packages '''
import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np 

# Load Data
data = sio.loadmat('data.mat')
x = data['x'].reshape([-1, 1])
y = data['y'].reshape([-1, 1])

# Plot the data using plot function
plt.plot(x, y)
plt.grid()

# Create the matrix
X = np.hstack((np.ones((len(x),1)), np.power(x,1)))

# Compute the least square line over the given data
X_t = X.transpose((1,0))
sol = np.dot(np.linalg.inv(np.dot(X_t,X)), np.dot(X_t,y))

# print "Assignment 2: "
# print sol

# Overlay the computed least square line over the given data
plt.hold(True)
plt.plot(x, sol[0] + sol[1]*x)

# Assign a title to this figure
plt.title('Least square line fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()