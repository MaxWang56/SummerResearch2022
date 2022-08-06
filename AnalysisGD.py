import numpy
from scipy.stats import ortho_group
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import random

# Covariance matrix
dim = 1000
x = ortho_group.rvs(dim)
x_t = np.transpose(x)

d = [[0 for j in range(len(x[0]))] for i in range(len(x))]
for i in range(len(x)):
    d[i][i] = math.e ** (-(i + 1))

covariance = np.matmul(d, x_t)
covariance = np.matmul(x, covariance)
eig_vals, eig_vecs = numpy.linalg.eig(covariance)

# Creating data samples
numDataPoints = 1000
mean = [0 for i in range(dim)]
data_list = np.random.multivariate_normal(mean, covariance, numDataPoints)
for i in range(numDataPoints):
    original_array = data_list[i]
    new_array = [0 for i in range(dim)]

    # Decomposing the data vectors into eigenbases and scaling each component by respective eigenvector
    for j in range(dim):
        eigenval_projection = np.matmul(np.transpose(eig_vecs[j]), data_list[i])
        eigenval_scaling = eigenval_projection * eig_vals[j]
        eigenvec_scaled = abs(eigenval_scaling) * eig_vecs[j]
        new_array = numpy.add(new_array, abs(eigenvec_scaled))
    data_list[i] = new_array

# Simple linear regression

# weights
w = np.array([random.uniform(0.1, 1) for i in range(dim)])
b = 0

# learning rate
learningRate = 0.1

# Weight progression
weightProgress = np.array([])

lossList = np.array([])
# Loss function
epochs = 100
count = 0
losslist = np.array([])
while count < epochs:
    if count == 0:
        weightProgress = w
    else:
        weightProgress = np.vstack([weightProgress, w])
    # Compute loss, L2 norm with target value and predicted
    loss = 0
    w_gradient = 0
    b_gradient = 0
    for i in data_list:
        prediction = np.dot(w, i) + b
        target = np.sum(i)
        difference = target - prediction
        loss += difference ** 2

        # Compute w gradient:
        w_gradient = w_gradient + (-2) * w * (np.sum(i) - (prediction))
        b_gradient = b_gradient + (-2) * (target - (prediction))
    losslist = np.append(losslist, loss)
    w_gradient = w_gradient / len(data_list)
    b_gradient = b_gradient / len(data_list)

    lossList = np.append(lossList, loss)

    # Update Weights
    w = w - learningRate * w_gradient
    b = b - learningRate * b_gradient

    count += 1
print(losslist)
# Eigenvectors of covariance matrix of data
datalistT = np.transpose(data_list)
dataCovar = np.matmul(datalistT, data_list)

w, v = LA.eig(dataCovar)
# magnitude of eigenvalues
w2 = [0 for i in range(len(w))]
for i in range(len(w)):
    eigenval = abs(w[i])
    w2[i] = eigenval
w = w2

w, v = LA.eig(covariance)

projectionList = np.array([])
for i in range(len(weightProgress)):
    weights = weightProgress[i]
    projection = np.array([])
    for j in range(len(v)):
        weightsT = np.transpose(weights / np.linalg.norm(weights))
        val = np.matmul(v[j], weightsT)
        val = abs(val)
        projection = np.append(projection, val)
    if i == 0:
        projectionList = projection
    else:
        projectionList = np.vstack([projectionList, projection])

x_axis = np.array([])
eigen_y_axis_1 = np.array([])
eigen_y_axis_2 = np.array([])
eigen_y_axis_3 = np.array([])
eigen_y_axis_4 = np.array([])
eigen_y_axis_5 = np.array([])
eigen_y_axis_10 = np.array([])

for i in range(dim):
    eigen_y_axis_1 = np.append(eigen_y_axis_1, projectionList[0][i])
    eigen_y_axis_2 = np.append(eigen_y_axis_2, projectionList[1][i])
    eigen_y_axis_3 = np.append(eigen_y_axis_3, projectionList[2][i])
    eigen_y_axis_4 = np.append(eigen_y_axis_4, projectionList[3][i])
    eigen_y_axis_5 = np.append(eigen_y_axis_5, projectionList[4][i])
    eigen_y_axis_10 = np.append(eigen_y_axis_10, projectionList[9][i])

    x_axis = np.append(x_axis, i)
plt.plot(x_axis, eigen_y_axis_1)
plt.xlabel("ith Eigen Value")
plt.ylabel("Projection Value")
plt.title("Weights projected onto Eigenvectors, Epoch 1")
plt.show()
plt.plot(x_axis, eigen_y_axis_2)
plt.xlabel("ith Eigen Value")
plt.ylabel("Projection Value")
plt.title("Weights projected onto Eigenvectors, Epoch 2")
plt.show()
plt.plot(x_axis, eigen_y_axis_3)
plt.xlabel("ith Eigen Value")
plt.ylabel("Projection Value")
plt.title("Weights projected onto Eigenvectors, Epoch 3")
plt.show()
plt.plot(x_axis, eigen_y_axis_4)
plt.xlabel("ith Eigen Value")
plt.ylabel("Projection Value")
plt.title("Weights projected onto Eigenvectors, Epoch 4")
plt.show()
plt.plot(x_axis, eigen_y_axis_5)
plt.xlabel("ith Eigen Value")
plt.ylabel("Projection Value")
plt.title("Weights projected onto Eigenvectors, Epoch 5")
plt.show()
plt.plot(x_axis, eigen_y_axis_10)
plt.xlabel("ith Eigen Value")
plt.ylabel("Projection Value")
plt.title("Weights projected onto Eigenvectors, Epoch 10")
plt.show()