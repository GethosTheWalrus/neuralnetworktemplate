import numpy
import scipy.special
import matplotlib.pyplot

# neural network class
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output
        final_outputs = self.activation_function(final_inputs)

        # calculate output layer error
        output_errors = targets - final_outputs

        # calculate hidden layer error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weights for links between hidden and output
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update weights for links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# input_nodes = 784
# hidden_nodes = 100
# output_nodes = 10

input_nodes = 2
hidden_nodes = 10
output_nodes = 2

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# data_file = open("mnist_train.csv", 'r')
# data_list = data_file.readlines()
# data_file.close()

# for record in data_list:
#     all_values = record.split(',')
#     # scaling the data to be  between 0 and 1
#     inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#     targets = numpy.zeros(output_nodes) + 0.01
#     targets[int(all_values[0])] = 0.99
#     n.train(inputs, targets)

# test_data_file = open("data_10.csv", 'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()

# all_values = test_data_list[7].split(',')

# print(all_values)
# print(n.query([numpy.asfarray(all_values[1:])]))

xor_possibilities = [
    [0,0,0],
    [1,0,1],
    [1,1,0],
    [0,1,1]
]

for i in range(0, 5000):
    index = numpy.random.randint(4)
    record = xor_possibilities[index]
    # scaling the data to be between 0 and 1
    inputs = (numpy.asfarray(record[1:]) / 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(record[0])] = 0.99
    n.train(inputs, targets)

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