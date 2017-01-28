from os import getcwd, walk
from os.path import join, split
import sys
import traceback

from tracer import Tracer

import matplotlib.pyplot as plt
from skimage.io import imshow, show

from property_loader import load_properties
from ruler import Ruler, MeasurementError
from species_sort import Species

"""
main2.py: The main file for Deciduous Tree Leaf Identification with Artificial Neural Networks.

Version 2 because of a major reworking in how images and properties are loaded.
Ignore the __version__ docstring as of now
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "8/14/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

# global variables
IMAGE_DIR = join(getcwd(), 'input-images')  # master image directory
DEFAULT_RULER = Ruler()

# create tracer
_f = __file__
tc = Tracer(mode=5)


"""
Classes for all species known to the program are added here.

Template species here:
# Test_leaf = Species('Test', 'leaf', properties=[
#     'simple',
#     'broad',
#     'ovate',
# ])

Directory structure:
input-images
--->Leaf Collection %date%
------->%genus% %species%
----------->%image name%.jpg

Properties of leaves are included to hopefully provide some insights
as to why the ANN (will envitably) confuse species.

They are loaded from a properties.txt file in the project folder.

Load locations (images known to be the aforementioned species)
are added for each of the species. The images in the directories given
to the species are used in training the ANN after the images are
measured.
"""


def load_species_structured(master_dir):
    """
    load all species found by walking through the image_dir
    :return: dictionary of bin_nom:species class
    """

    # find all leaves by walking through the leaf collections
    tc.print('Finding all leaves by walking through the leaf collections', 4, _f)
    species_list = []
    for root, paths, filenames in walk(master_dir):
        if root.find('Leaf Collection') > -1:   # looks for leaf collections, if str is present
                                                # it will have an index
            # it must be determined whether we are in the collection only or in a species sub folder
            if not filenames:  # in collection
                species_list.extend(paths)
                # elif not paths: # in species
                #     print(filenames)

    tc.print(species_list, 3, _f)

    # only get the species present, no repeats
    # if a species is repeated or called 'unsorted', do not load it
    all_s_species = list(set(species_list))
    all_s_species.remove('unsorted')  # removes the unsort images
    species_dict = {}

    # load the species properties for the given g, s
    for species_str in all_s_species:
        gen, spe = species_str.split(' ')
        species_dict[species_str] = Species(genus=gen,
                                            species=spe,
                                            properties=load_properties(species_str))

    # recursively walk through the image-dir and record load locations to the species in the dict
    for root, paths, filenames in walk(master_dir):
        if 'Leaf Collection' in split(root)[1]:
            for species_str in paths:
                if species_str != 'unsorted':
                    full_p = join(root, species_str)

                    tc.print('added {0} as load location for {1}'.format(split(split(root)[0])[1], species_str), 4, _f)
                    
                    species_dict[species_str].add_load_location(full_p, split(split(root)[0])[1])

    # print all species present
    tc.print('\nSpecies loaded: ', 4, _f)
    for k in sorted(species_dict.keys()):
        tc.print(str(species_dict[k]), 4, _f)

    # append all values of dict to list
    species_output = [species_dict[k] for k in species_dict]

    # return a list of values from species_dict
    return species_output


def use_ruler(species, ruler=Ruler()):
    """
    THIS IS A DEMONSTRATION/VISUALIZATION OF THE RULER IN USE

    Loads species specified and measures the species with
    Ruler(). Plots and displays the dfferent leaf images.
    :param species:
    :param ruler:
    :return:
    """

    # state current leaf species and localize path
    tc.print((species.bin_nom, species.load_locations, species.get_leaf_paths()[0]), 4, _f)
    leaf_path = species.get_leaf_paths()[0]

    # load the leaf with the ruler
    ruler.load_new_image(leaf_path)

    # img, lines, lines2, lines3, length, center_range
    img = ruler.leaf
    hough_center = ruler.vein_measure['hough center']
    hough_above = ruler.vein_measure['hough above']
    hough_below = ruler.vein_measure['hough below']
    hough_range = ruler.vein_measure['center range']
    midrib_line = ruler.vein_measure['midrib lin approx']
    length = ruler.length

    print(hough_above)

    # displaying data with pyplot and matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(5, 2))
    ax = axes.ravel()
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('{0}, {1}cm'.format(species.bin_nom, length))
    row, col = img.shape
    ax[1].axis((0, col, row, 0))
    ax[1].imshow(-img, cmap=plt.cm.gray)
    for line in hough_center:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'b')
    for line in hough_above:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'g')
    for line in hough_below:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'r')
    ax[1].plot((0, img.shape[1]), (hough_range[0], hough_range[0]), 'b--')
    ax[1].plot((0, img.shape[1]), (hough_range[1], hough_range[1]), 'g--')
    ax[1].plot((0, img.shape[1]), (midrib_line(0), midrib_line(img.shape[1])))
    ax[2].imshow(ruler.leaf_bin, cmap=plt.cm.gray)
    ax[2].set_title('{0} binary'.format(species.bin_nom))
    ax[3].imshow(ruler.scale_bin, cmap=plt.cm.gray)
    ax[3].set_title('{0} scale'.format(species.bin_nom))

    plt.show()


# load all of the leaves found
def measure_leaves_list(species_list, r=DEFAULT_RULER, no_scale=False):
    """
    Measure all leaves from list of species, given from load_species_structured.

    Should theoretically be able to accept any list of leaves, given that it is strucuted
    in a list (hence the name).

    :param no_scale:
    :param r: ruler
    :param species_list: list of all of the species loaded by load_species_structured
    :return: number of successes and failures from measuring leaves
    """

    # counter for successful and unsuccessful measurements
    successes = 0
    fails = 0

    for species in species_list:
        for leaf_path in species.get_leaf_paths():
            try:
                print('')
                tc.print('Loading and measuring {0} \n@ {1}'.format(species.bin_nom, leaf_path), 5, _f)
                # noinspection PyBroadException
                try:
                    r.load_new_image(leaf_path, no_scale=no_scale)
                    tc.print("Loaded. Saving data to ruler...", 5, _f)

                    r.save_data(species.bin_nom)
                    tc.print('Saved.', 5, _f)

                    successes += 1

                except MeasurementError as e:
                    print(e)
                    tc.print("Error: Failed to measure {0}".format(species.bin_nom), 1, _f)
                    tb = sys.exc_info()[2]
                    traceback.print_tb(tb)

                    fails += 1

                except OSError as e:
                    print(e)
                    tb = sys.exc_info()[2]
                    traceback.print_tb(tb)

                    fails += 1

            except TypeError as e:
                tc.print('Error encountered at leaf path {0}:'.format(leaf_path), 1, _f)
                tb = sys.exc_info()[2]
                traceback.print_tb(tb)

                fails += 1

    return successes, fails


# start main
if __name__ == '__main__':
    # create ruler
    r = Ruler()

    # use_ruler(Liriodendron_tulipifera, r)
    # print("saving data to ruler")
    # r.save_data(Liriodendron_tulipifera.bin_nom)

    # load the leaves from the harddrive
    leaves_list = load_species_structured(IMAGE_DIR)

    # measure the found leaves
    s, f = measure_leaves_list(leaves_list)

    tc.print('\n\nFINSIHED\n{0} leaves measured successfully and {1} failed attempts.'.format(s, f), 4, _f)
