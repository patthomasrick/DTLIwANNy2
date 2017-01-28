import fann2.libfann as fann

"""
ann-manager.py: Code to provide methods to create and control an ANN with libfann.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "12/12/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"


# global vars
CONNECTION_RATE = 1.0   # connectivity of ANN, 1 is complete, 0.5 is half connected
LEARNING_RATE = 0.7     # how fast the ANN learns the training data
DESIRED_ERROR = 0.005
MAX_ITERATIONS = 5000   # cut off to number of training cycles
ITERATIONS_BETWEEN_REPORTS = 1000

DEFAULT_ANN_PATH = 'ann/neural-net.ann'


class NeuralNetwork:
    """
    The class that contains contrls for ANN
    """
    def __init__(self):
        """
        Initialize the ANN's variable
        """

        self.ann = fann.neural_net()

    def load_ann_from_file(self, nn_file):
        """
        Loads a saved ANN from a file. The file is created by FANN.
        :param nn_file: path to file
        :return: None
        """

        self.ann.create_from_file(nn_file)

    def train(self,
              training_file_path,
              num_inputs,
              num_outputs,
              nn_path=DEFAULT_ANN_PATH,
              num_hid_neurons=None):
        """
        Trains an ANN from data containing in a text-based training file that is created by xml2trainingdata.py.
        :param two_hid: specifies whether to use two hidden layers or not (default not)
        :param training_file_path: path to training file
        :param nn_path: path to save ANN to
        :param num_inputs: number of input neurons
        :param num_outputs: number of output neurons
        :param num_hid_neurons: number of hidden neurons
        :return: None
        """

        # if hidden neurons are not specified, set the number to 2/3 of the sum of input and output neurons
        if num_hid_neurons is None:
            num_hid_neurons = (2*(num_inputs * num_outputs))/3

        ann_tuple = (num_inputs, num_hid_neurons, num_outputs)

        # create the ANN
        ann = fann.neural_net()
        ann.create_sparse_array(CONNECTION_RATE, ann_tuple)

        # set learning style
        ann.set_learning_rate(LEARNING_RATE)

        # set activation function
        ann.set_activation_function_output(fann.SIGMOID_SYMMETRIC)

        # train the ANN on file
        ann.train_on_file(training_file_path,
                          MAX_ITERATIONS,
                          ITERATIONS_BETWEEN_REPORTS,
                          DESIRED_ERROR)

        # save ann to file and free memory associated with it
        ann.save(nn_path)
        ann.destroy()

        # set own ann to saved ann
        self.load_ann_from_file(nn_path)

    def run(self, numbers, species=None):
        """
        Runs the ann on numbers given. This is how the ANN is supposed to be used.
        :param numbers: list of numbers representing a leaf
        :param species: a list of all species possible to map the output to
        :return: outputs corresponding to species
        """

        # get the ANN's output
        if species is None:
            species = []
        output = self.ann.run(numbers)

        # return
        if len(species) == len(output):
            dict_output = {}
            for i, s in enumerate(sorted(species)):
                dict_output[s] = output[i]

            # return dict output
            return dict_output

        else:  # unsorted
            return output
