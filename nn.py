import numpy
import scipy.special
import codecs, json

# neural network class
class neuralNetwork:
    def __init__(self):
        # read the parameters from the json file
        jsonString = codecs.open("params.json", 'r', encoding='utf-8').read()
        self.params = json.loads(jsonString)

        # set params from json
        self.inodes = self.params["input_nodes"]
        self.hnodes = self.params["hidden_nodes"]
        self.onodes = self.params["output_nodes"]
        self.lr = self.params["learning_rate"]

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

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

    def loadconfig(self):
        # set weights from json
        self.wih = numpy.array(self.params["weights"]["xor"]["wih"])
        self.who = numpy.array(self.params["weights"]["xor"]["who"])

    def saveconfig(self):
        # save params to the file
        params = {
            "input_nodes": self.inodes,
            "output_nodes": self.onodes,
            "hidden_nodes": self.hnodes,
            "learning_rate": self.lr,
            "weights": {
                "xor": {
                    "wih": self.wih,
                    "who": self.who
                }
            }
        }
        f = open("params.json", "w")
        f.write(json.dumps(params, cls=NumpyEncoder))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)