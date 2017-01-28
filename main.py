from os import getcwd, walk
from os.path import join

import matplotlib.pyplot as plt

import scale_reader
from ruler import Ruler
from species_sort import Species

# ~/.pyenv/versions/3.5.2/bin/python

"""
main.py: The main file for Deciduous Tree Leaf Identification with Artificial Neural Networks.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "6/28/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

# leaf image paths
COL1_PATH = join(getcwd(), 'input-images', 'Leaf Collection 160703')
COL2_PATH = join(getcwd(), 'input-images', 'Leaf Collection 160711')

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
Acer_nigrum = Species('Acer', 'nigrum', properties=[
    'simple',
    'lobed',
    'simple',
    'closely related to Acer saccharum',
])
Acer_pensylvanicum = Species('Acer', 'pensylvanicum', properties=[
    'simple',
    'lobed',
    'three lobes',
    'palmatedly lobed',
    'palmately veined',
    'toothed',
])
Acer_saccharum = Species('Acer', 'saccharum', properties=[
    'simple',
    'lobed',
    'palmatedly lobed',
    'palmately veined',
    'deep sinues',
])
Carpinus_caroliniana = Species('Carpinus', 'caroliniana', properties=[
    'simple',
    'toothed',
    'pinnately veined',
    'ovate',
    'broad',
])
Cercus_canadensis = Species('Cercis', 'canadensis', properties=[
    'simple',
    'heart shaped',
    'broad',
    'ovate',
])
Liriodendron_tulipifera = Species('Liriodendron', 'tulipifera', properties=[
    'simple',
    'lobed'
    'four lobes',
    'heart shaped',
])
Quercus_alba = Species('Quercus', 'alba', properties=[
    'simple',
    'obovate',
    'oblong',
    'lobed',
    'seven to nine lobes',
])
Quercus_coccinea = Species('Quercus', 'coccinea', properties=[
    'simple',
    'broad',
    'lobed',
    'seven lobes',
    'deep sinuses',
])
# Quercus_unknown = Species('Quercus', 'unknown', properties=[
#     'simple',
#     'lobed',
#     'broad',
# ])
Rhododendron_calendulaceum = Species('Rhododendron', 'calendulaceum', properties=[
    'villous',
    'broad',
    'ovate',
    'pinnately veined',
    'woody shrub',
])

Tilla_americana = Species('Tilla', 'americana', properties=[
    'broad',
    'simple',
    'ovate',
    'pinnately veined',
    'toothed',
])

# .add_load_location(, 'Col #')
Acer_nigrum.add_load_location(
    join(COL1_PATH, 'Acer nigrum'), 'Col 1')

Acer_pensylvanicum.add_load_location(
    join(COL2_PATH, 'Acer pensylvanicum'), 'Col 2')

Acer_saccharum.add_load_location(
    join(COL2_PATH, 'Acer saccharum'), 'Col 2')

Carpinus_caroliniana.add_load_location(
    join(COL2_PATH, 'Carpinus caroliniana subsp. virginiana'), 'Col 2')

Cercus_canadensis.add_load_location(
    join(COL1_PATH, 'Cercis canadensis'), 'Col 1')

Liriodendron_tulipifera.add_load_location(
    join(COL1_PATH, 'Liriodendron tulipifera'), 'Col 1')

Quercus_alba.add_load_location(
    join(COL1_PATH, 'Quercus alba'), 'Col 1')
Quercus_alba.add_load_location(
    join(COL2_PATH, 'Quercus alba'), 'Col 2')

# Quercus_coccinea.add_load_location(
#     join(COL1_PATH, 'Quercus coccinea'), 'Col 1')
#
# Quercus_unknown.add_load_location(
#     join(COL1_PATH, 'Unknown Quercus'), 'Col 1')

Rhododendron_calendulaceum.add_load_location(
    join(COL2_PATH, 'Rhododendron calendulaceum'), 'Col 2')

# Test_leaf.add_load_location(
#     join(COL1_PATH, 'Test Leaf'), 'Col test')

Tilla_americana.add_load_location(
    join(COL2_PATH, 'Tilla americana'), 'Col 2')

ALL_SPECIES = [
    Acer_nigrum,
    Acer_saccharum,
    Acer_pensylvanicum,
    Carpinus_caroliniana,
    Cercus_canadensis,
    Liriodendron_tulipifera,
    Quercus_alba,
    # Quercus_coccinea,
    # Quercus_unknown,
    # Test_leaf,
    Rhododendron_calendulaceum,
]

"""
The later functions are different tests used to test the
functionality of the ruler.py functions and classes. Functionality
of ruler.py has changed over time, thus possibly rendering some of
the earlier function tests useless.
"""


def load_all_leaves():
    """
    Walk through the directory tree to find species and images
    :return:
    """

def display_all_leaves(all_species):
    """
    Displays all leaves in all_species.
    Subplot size limits this.
    :return: None
    """
    subplot_size = (2, 3)

    path_dict = {}
    for species in all_species:
        print(species.bin_nom, species.load_locations, species.get_leaf_paths()[0])
        path_dict[str(species)] = species.get_leaf_paths()[0]

    fig, axes = plt.subplots(subplot_size[0], subplot_size[1], figsize=(3, 2))
    ax = axes.ravel()
    for count, path_key in enumerate(path_dict.keys()):
        img, length = scale_reader.import_leaf_image(path_dict[path_key])
        ax[count].imshow(-img, cmap=plt.cm.gray)
        ax[count].set_title("{0} {1}cm".format(path_key, length))
    plt.show()


def display_hough_lines():
    path_dict = {}
    for species in ALL_SPECIES:
        print(species.bin_nom, species.load_locations, species.get_leaf_paths()[0])
        path_dict[str(species)] = species.get_leaf_paths()[0]

    saved = []

    fig, axes = plt.subplots(2, 3, figsize=(3, 2))
    ax = axes.ravel()
    for count, path_key in enumerate(path_dict.keys()):
        img, lines, length = scale_reader.import_leaf_image(path_dict[path_key])
        saved.append([img, lines])
        ax[count].imshow(-img, cmap=plt.cm.gray)
        ax[count].set_title("{0} {1}cm".format(path_key, length))
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(3, 2))
    ax = axes.ravel()

    for count, pair in enumerate(saved):
        img, lines = pair
        ax[count].imshow(-img, cmap=plt.cm.gray)
        for line in lines:
            p0, p1 = line
            ax[count].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[count].set_title("Hough")
    plt.show()


def compare_hough_lines(species):
    """
    Display a leaf alongside the the hough probablistic line transformation
    :param species: the species to be transformed
    :return: None
    """
    print(species.bin_nom, species.load_locations, species.get_leaf_paths()[0])
    leaf_path = species.get_leaf_paths()[0]
    img, lines, lines2, lines3, length, center_range = scale_reader.import_leaf_image(leaf_path)

    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    ax = axes.ravel()
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('{0}, {1}cm'.format(species.bin_nom, length))
    row, col = img.shape
    ax[1].axis((0, col, row, 0))
    ax[1].imshow(-img, cmap=plt.cm.gray)
    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'b')
    for line in lines2:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'g')
    for line in lines3:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'r')
    print(center_range)
    ax[1].plot((0, img.shape[0]), (center_range[0], center_range[0]), 'b--')
    ax[1].plot((0, img.shape[0]), (center_range[1], center_range[1]), 'g--')
    plt.show()
    return None


def use_ruler(species, ruler=Ruler()):
    """
    Loads species specified and measures the species with
    Ruler(). Plots and displays the dfferent leaf images.
    :param species:
    :param ruler:
    :return:
    """
    print(species.bin_nom, species.load_locations, species.get_leaf_paths()[0])
    leaf_path = species.get_leaf_paths()[0]

    ruler.load_new_image(leaf_path)

    # img, lines, lines2, lines3, length, center_range
    img = ruler.leaf
    hough_center = ruler.vein_measure['hough center']
    hough_above = ruler.vein_measure['hough above']
    hough_below = ruler.vein_measure['hough below']
    hough_range = ruler.vein_measure['center range']
    midrib_line = ruler.vein_measure['midrib line']
    length = ruler.length

    # displaying data
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


def measure_all_leaves():
    for s in ALL_SPECIES:
        for p in s.get_leaf_paths():
            try:
                print('loading and measuring', s.bin_nom, 'at', p)
                r.load_new_image(p)
                print("saving data to ruler")
                r.save_data(s.bin_nom)
            except TypeError as e:
                print(e)

if __name__ == '__main__':
    r = Ruler()
    # use_ruler(Liriodendron_tulipifera, r)
    # print("saving data to ruler")
    # r.save_data(Liriodendron_tulipifera.bin_nom)

    measure_all_leaves()
