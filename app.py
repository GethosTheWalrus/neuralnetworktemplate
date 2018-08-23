import numpy
from nn import neuralNetwork
import codecs, json

# create the network object
n = neuralNetwork()

# load the previously generated weights
n.loadconfig()

xor_possibilities = [
    [0,0,0],
    [1,0,1],
    [1,1,0],
    [0,1,1]
]

# train the network
for i in range(0, 100000):
    index = numpy.random.randint(4)
    record = xor_possibilities[index]
    # scaling the data to be between 0 and 1
    inputs = (numpy.asfarray(record[1:]) / 0.99) + 0.01
    targets = numpy.zeros(n.onodes) + 0.01
    targets[int(record[0])] = 0.99
    n.train(inputs, targets)

# save the updated weights
n.saveconfig()

# predict the output of the 4 xor possiblities
xor_00 = n.query( [ numpy.asfarray( xor_possibilities[0][1:] ) ] )
xor_01 = n.query( [ numpy.asfarray( xor_possibilities[1][1:] ) ] )
xor_10 = n.query( [ numpy.asfarray( xor_possibilities[2][1:] ) ] )
xor_11 = n.query( [ numpy.asfarray( xor_possibilities[3][1:] ) ] )

print("0 xor 0 prediction: ", xor_00)
print("--------------------------------------------")
print("0 xor 1 prediction: ", xor_01)
print("--------------------------------------------")
print("1 xor 0 prediction: ", xor_10)
print("--------------------------------------------")
print("1 xor 1 prediction: ", xor_11)