# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
...
# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8] #First 7 columns
y = dataset[:,8] #Last column as a output
#print (X)
#print (y)
...
# define the keras model
model = Sequential() 
# Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

#Creating a Sequential model incrementally via the add() method.
#Fully connected layers are defined using the "Dense class".
model.add(Dense(12, input_shape=(8,), activation='relu')) #First hidden layer with 12 nodes and an input of 8 nodes, activation function of ReLu
#NOTE: first Dense layer is doing two things, defining the input or visible layer and the first hidden layer.

model.add(Dense(8, activation='relu')) #Second hidden layer with 8 nodes and activation function of ReLu
model.add(Dense(1, activation='sigmoid')) #One Output layer with the activation of sigmoid 
...
#compile the keras model: specify properties required when training the network
#training a network means "finding the best set of weights" to map inputs to outputs

#specify the loss function to use to evaluate a set of weights, use cross entropy as the loss argument for binary classification problems
# It is defined in Keras as "binary_crossentropy".

#the optimizer used to search through different weights for the network, the efficient stochastic gradient descent algorithm “adam“
#It is a popular version of gradient descent because it automatically tunes itself
#This gradient descent algorithm gives good results in a wide range of problems

#and any optional metrics you want to collect and report during training. For this case we´ll use "accuracy" 
#To collect and report the classification accuracy defined "via the metrics" argument.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
...
