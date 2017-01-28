from sys import exit
from os.path import join, split, isdir
from os import listdir, walk
from os import remove as osremove

from annmanager import *
from main2 import *
from ruler import Ruler
from xml2trainingdata import *
from output_to_xlsx import *

"""
main3.py: Code to measure leaves, save data, possibly view data, and run ANNs. Should ideally be able to produce
results.

Has most of the functionality of main2.py, dataviewer.py, and some more functions not included elsewhere. Provides a
basic menu system for the user to navigate. Mainly intended for the researcher's use, however, it is somewhat simple
to use (prior knowledge is needed, however, and documentation is nonexistant.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "12/12/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

# global variables
IMAGE_DIR = join(getcwd(), 'input-images')  # master image directory
ANN = NeuralNetwork()

MAX_SPECIES_SET_REPS = 100

# create tracer
tc = Tracer(mode=5)

# functions for each option
def view_leaf():
    """
    12/13/16

    Loads a leaf and displays it with matplotlib.
    :return: None
    """

    return None


def measure_all_leaves():
    """
    12/13/16

    Measures all leaves in all collections.
    COPY AND PASTED FROM MAIN2.PY
    :return: number of successes and failures
    """

    # remove the old measurements
    try:
        osremove('save-data/leaf-data.xml')
    except FileNotFoundError:
        pass

    # # create ruler
    # ruler = Ruler()

    # load the leaves from the harddrive
    leaves = load_species_structured(IMAGE_DIR)

    # measure the found leaves
    successes, failures = measure_leaves_list(leaves)

    # report successes and failures
    print('\n\nFINSIHED\n{0} leaves measured successfully and {1} failed attempts.'.format(successes, failures))
    return successes, failures


def load_year_1_leaves():
    """
    1/10/17

    Load all of the species in the specific folder for year 1 leaves. A modified verison of what exists in main2.py.
    :return: dictionary of bin_nom:species class
    """

    # find all leaves by walking through the leaf collections
    species_list = []
    for root, paths, filenames in walk(IMAGE_DIR):
        if root.find('Year 1 Images') > -1:  # looks for leaf collections, if str is present
            # it will have an index
            # it must be determined whether we are in the collection only or in a species sub folder
            if not filenames:  # in collection
                species_list.extend(paths)
                # elif not paths: # in species
                #     print(filenames)

    print(species_list)

    # only get the species present, no repeats
    # if a species is repeated or called 'unsorted', do not load it
    all_s_species = list(set(species_list))
    # all_s_species.remove('unsorted')  # removes the unsort images
    species_dict = {}

    # load the species properties for the given g, s
    for species_str in all_s_species:
        gen, spe = species_str.split(' ')
        species_dict[species_str] = Species(genus=gen,
                                            species=spe,
                                            #properties=load_properties(species_str)
        )

    # recursively walk through the image-dir and record load locations to the species in the dict
    for root, paths, filenames in walk(IMAGE_DIR):
        if 'Year 1 Images' in split(root)[1]:
            for species_str in paths:
                if species_str != 'unsorted':
                    full_p = join(root, species_str)

                    print('added {0} as load location for {1}'.format(split(split(root)[0])[1], species_str))
                    species_dict[species_str].add_load_location(full_p, split(split(root)[0])[1])

    # print all species present
    print('\nSpecies loaded:')
    for k in sorted(species_dict.keys()):
        print(str(species_dict[k]))

    # append all values of dict to list
    species_output = [species_dict[k] for k in species_dict]

    print(species_output)

    # return a list of values from species_dict
    return species_output


def measure_yr1_leaves():
    """
    1/10/17

    Same as measure leaves.

    :return:
    """

    leaves = load_year_1_leaves()
    successes, failures = measure_leaves_list(leaves, r=DEFAULT_RULER, no_scale=True)

    print('\n\nFINSIHED\n{0} leaves measured successfully and {1} failed attempts.'.format(successes, failures))
    return successes, failures


def run_dataviewer():
    """
    12/13/16

    Runs the code in dataviewer.py.
    Nothing is imported from the code, as the code itself is directly executed.
    :return: None
    """
    # opens dataviewer and executes its code directly
    with open("dataviewer.py", 'r') as dataviewer_file:
        code = compile(dataviewer_file.read(), "dataviewer.py", 'exec')
        exec(code)

    return None


def convert_xml_train():
    """
    12/13/16

    Uses the tools in xml2trainingdata.py to convert the XML file generated by the leaf measurer to FANN training data.
    :return: None
    """
    convert(DEFAULT_XML, DEFAULT_TRAINING_FILE)
    return None


def train_ann():
    """
    12/13/16

    Trains the artificial neural network to the sample of training leaves from convert_xml_train().
    :return: None
    """
    # read the first line of the training file
    with open(DEFAULT_TRAINING_FILE, 'r') as train_file:
        header = train_file.readline()
        train_file.close()
    total, num_input, num_output = header.split(' ')
    num_input = int(num_input)
    num_output = int(num_output)

    # train ANN
    ANN.train(DEFAULT_TRAINING_FILE, num_input, num_output)

    return None


def run_ann():
    """
    12/13/16

    Runs the ANN on the measurements of the leaves not in the training sample (the run sample).
    :return: None
    """
    ANN.load_ann_from_file(DEFAULT_ANN_PATH)

    all_leaves, train_leaves, run_leaves = load_leaf_samples(DEFAULT_SAMPLE_SAVE_FILE)

    outputs = []

    for species in train_leaves.keys():
        leaf_list = train_leaves[species]

        for leaf in leaf_list:
            # ############## INPUT LINE #################

            #
            # order of data in training file
            #
            # perimeter
            # length
            # area
            # surface variability mean
            # surface variability median
            # vein angles above
            # vein angles below
            # vein length
            # contour angles
            # contour positions
            # contour sizes

            # create line
            line = [
                leaf['p'],
                leaf['length'],
                leaf['area'],
                leaf['sf_v_mean'],
                leaf['sf_v_median'],
                leaf['vein_angle_above'],
                leaf['vein_angle_below'],
                leaf['vein_length']
            ]

            # extend line with contour information
            line.extend(leaf['contour_angle'].values())
            # line.extend(leaf['contour_pos'])
            line.extend(leaf['contour_size'].values())

            output ={
                'species': species,
                'leaf': leaf,
                'ann': ANN.run(line, species=species)
            }

            outputs.append(output)

    save_ann_output_to_excel(DEFAULT_WB_FILENAME, outputs)

    return outputs


# noinspection PyNoneFunctionAssignment
def convert_train_run():
    """
    12/13/16

    Basically a run of the almost complete package: convert, train, then run.
    :return: combined outputs of 3 functions
    """
    out1 = convert_xml_train()
    out2 = train_ann()
    out3 = run_ann()

    return out1, out2, out3


# noinspection PyNoneFunctionAssignment
def measure_convert_train_run():
    """
    12/13/16

    Basically a run of the complete package: measure, convert, train, then run.
    :return: combined outputs of 4 functions
    """
    out1 = convert_xml_train()
    out2 = train_ann()
    out3 = run_ann()
    out4 = convert_train_run()

    return out1, out2, out3, out4


def convert_train_run_repeat():
    """
    12/13/16

    Basically a run of the almost complete package on repeat: convert, train, then run.
    :return: combined outputs of 3 functions
    """
    n = input('Number of repitions? -> ')

    try:
        n = int(n)
        for i in range(0, n):
            convert_train_run()

    except ValueError:
        print('The number you entered is not valid.')


def full_test():
    """
    1/6/17

    Runs a test on all the leaves, starting with random pairs of 2 species and eventually heading to the maximum number
    of leaves. The number of random pairs is 30 and the maximum leaves depends on the minimum number of leaves allowed
    in a training data set.

    :return:
    """

    # get the number of valid species

    # first load the XML file
    leaf_data_dict = load_xml(DEFAULT_XML)

    local_leaf_data_dict = leaf_data_dict.copy()  # to prevent changing master leaf data dict

    # filter leaves by min_leaves as done in xml2trainingdata.py
    keys_to_remove = []
    for species in local_leaf_data_dict.keys():
        if len(local_leaf_data_dict[species]) < MIN_LEAVES:
            keys_to_remove.append(species)

    for key in keys_to_remove:
        local_leaf_data_dict.pop(key)

    # get the max number of species
    max_species = len(local_leaf_data_dict)
    print(max_species)

    # PRODUCE DATA
    # set up counter for saving
    last_row = 8

    # iterate through all possible numbers of leaves
    for i in range(2, max_species):
        print('Set of {0} leaves'.format(i))

        # coutner 2 resets between numbers of species
        last_row_2 = 8

        # take a random set of species
        for j in range(0, MAX_SPECIES_SET_REPS):
            print('Repition {0}'.format(j+1))
            # create a training data set and corresponding run leaves
            save_train(DEFAULT_TRAINING_FILE,
                       local_leaf_data_dict,
                       num_species=i,
                       min_leaves=MIN_LEAVES)

            # train the ANN on the newly created data set
            train_ann()

            # run the ANN
            # ### copy and pasted from run_ann
            ANN.load_ann_from_file(DEFAULT_ANN_PATH)

            all_leaves, train_leaves, run_leaves = load_leaf_samples(DEFAULT_SAMPLE_SAVE_FILE)

            outputs = []

            for species in run_leaves.keys():
                leaf_list = run_leaves[species]

                for leaf in leaf_list:
                    # ############## INPUT LINE #################

                    #
                    # order of data in training file
                    #
                    # perimeter
                    # length
                    # area
                    # surface variability mean
                    # surface variability median
                    # vein angles above
                    # vein angles below
                    # vein length
                    # contour angles
                    # contour positions
                    # contour sizes

                    # create line
                    line = [
                        leaf['p'],
                        leaf['length'],
                        leaf['area'],
                        leaf['sf_v_mean'],
                        leaf['sf_v_median'],
                        leaf['vein_angle_above'],
                        leaf['vein_angle_below'],
                        leaf['vein_length']
                    ]

                    # extend line with contour information
                    line.extend(leaf['contour_angle'].values())
                    # line.extend(leaf['contour_pos'])
                    line.extend(leaf['contour_size'].values())

                    output = {
                        'species': species,
                        'leaf': leaf,
                        'ann': ANN.run(line, species=species)
                    }

                    outputs.append(output)

            # save the data to a spreadsheet
            # create a worksheet name based on test ran
            ws_name = "{0} Species, Rep {1}".format(i, j+1)

            last_row, last_row_2 = save_ann_output_to_excel_one_file(
                DEFAULT_WB_FILENAME,
                ws_name,
                outputs,
                last_row,
                last_row_2)


# provide a simple text-based control system
if __name__ == '__main__':

    print(
        '''Patrick Thomas, 12/12/16

Deciduous Tree Leaf Identification with Artificial Neural Networks, Year 2

This project aims to provide the tools to identify a leaf of a given set of species.


'''
    )

    # dictionary of options
    choice_dict = {
        1: view_leaf,
        2: measure_all_leaves,
        3: run_dataviewer,
        4: convert_xml_train,
        5: train_ann,
        6: run_ann,
        7: convert_train_run,
        8: measure_convert_train_run,
        9: convert_train_run_repeat,
        10: full_test,
        11: measure_yr1_leaves,
        0: exit,
    }

    # try running from args first
    args = sys.argv
    try:
        # run arg
        choice = int(args[1])
        choice_dict[choice]()

        # exit if successful
        choice_dict[0]
    except IndexError:
        pass
    except KeyError:
        pass

    while True:
        # provide options and get user response
        print('\nPlease choose one of the following options:')

        print('1   View leaf')
        print('2   Measure all leaves')
        print('3   Use data-viewer')
        print('4   Convert XML data to training data')
        print('5   Train ANN, one hidden layer')
        print('6   Run ANN')
        print('7   Options 4, 5, & 6')
        print('8   Options 2, 4, 5, & 6')
        print('9   Repeat options 4, 5, & 6')
        print('10  Full Test on Currently Measured Leaves')
        print('11  Measure Year 1 Leaves Using Year 2 Methods')

        print('0 - Quit')
        choice = input('-> ')

        choice = int(choice)
        choice_dict[choice]()

        # try:
        #     choice = int(choice)
        # except ValueError:
        #     print('The number you entered is not valid.')
        #
        # try:
        #     choice_dict[choice]()
        # except KeyError:
        #     print('The number you entered is not a choice.')
