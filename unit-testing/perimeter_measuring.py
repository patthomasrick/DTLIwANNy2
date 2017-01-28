import sys
from os import getcwd, walk
from os.path import join, split

import matplotlib.pyplot as plt

sys.path.append('/home/patrick/PycharmProjects/SciFair-Y2')

from property_loader import load_properties
from ruler import Ruler
from species_sort import Species

# ~/.pyenv/versions/3.5.2/bin/python

"""
A modified version of the main file that only aims to measure perimeter
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "8/22/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

# leaf image paths
IMAGE_DIR = join('/home/patrick/PycharmProjects/SciFair-Y2', 'input-images')
PROP_DIR = join(IMAGE_DIR, 'properties')


def load_all_species():
    """
    load all species found by walking through the image_dir
    :return: dictionary of bin_nom:species class
    """
    species_list = ['Ostrya virginiana']
    # for root, paths, filenames in walk(IMAGE_DIR):
    #     if root.find('Leaf Collection') > -1:  # looks for leaf collections, if str is present it will have an index
    #         # it must be determined whether we are in the collection only or in a species sub folder
    #         if not filenames:  # in collection
    #             species_list.extend(paths)
    #             # elif not paths: # in species
    #             #     print(filenames)

    # only get the species present, no repeats
    all_s_species = list(set(species_list))
    #all_s_species.remove('unsorted')  # removes the unsort images
    species_dict = {}

    all_s_species = ['Ostrya virginiana']

    # load the species properties for the given g, s
    for species in all_s_species:
        g, s = species.split(' ')
        species_dict[species] = Species(genus=g,
                                        species=s,
                                        properties=load_properties(species, p=PROP_DIR))

    # recursively walk through the image-dir and record load locations to the species in the dict
    for root, paths, filenames in walk(IMAGE_DIR):
        if 'Leaf Collection' in split(root)[1]:
            for species in paths:
                if species != 'unsorted':
                    full_p = join(root, species)
                    print('added {0} as load location for {1}'.format(split(split(root)[0])[1], species))
                    species_dict[species].add_load_location(full_p, split(split(root)[0])[1])

    # print all species present
    print('Species loaded:')
    for k in species_dict:
        print(str(species_dict[k]))

    return species_dict.values()


# def use_ruler(species, ruler=Ruler()):
#     """
#     Loads species specified and measures the species with
#     Ruler(). Plots and displays the dfferent leaf images.
#     :param species:
#     :param ruler:
#     :return:
#     """
#     print(species.bin_nom, species.load_locations, species.get_leaf_paths()[0])
#     leaf_path = species.get_leaf_paths()[0]
#
#     ruler.load_new_image(leaf_path)
#
#     # img, lines, lines2, lines3, length, center_range
#     img = ruler.leaf
#     hough_center = ruler.vein_measure['hough center']
#     hough_above = ruler.vein_measure['hough above']
#     hough_below = ruler.vein_measure['hough below']
#     hough_range = ruler.vein_measure['center range']
#     midrib_line = ruler.vein_measure['midrib line']
#     length = ruler.length
#
#     # displaying data
#     fig, axes = plt.subplots(2, 2, figsize=(5, 2))
#     ax = axes.ravel()
#     ax[0].imshow(img, cmap=plt.cm.gray)
#     ax[0].set_title('{0}, {1}cm'.format(species.bin_nom, length))
#     row, col = img.shape
#     ax[1].axis((0, col, row, 0))
#     ax[1].imshow(-img, cmap=plt.cm.gray)
#     for line in hough_center:
#         p0, p1 = line
#         ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'b')
#     for line in hough_above:
#         p0, p1 = line
#         ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'g')
#     for line in hough_below:
#         p0, p1 = line
#         ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), 'r')
#     ax[1].plot((0, img.shape[1]), (hough_range[0], hough_range[0]), 'b--')
#     ax[1].plot((0, img.shape[1]), (hough_range[1], hough_range[1]), 'g--')
#     ax[1].plot((0, img.shape[1]), (midrib_line(0), midrib_line(img.shape[1])))
#     ax[2].imshow(ruler.leaf_bin, cmap=plt.cm.gray)
#     ax[2].set_title('{0} binary'.format(species.bin_nom))
#     ax[3].imshow(ruler.scale_bin, cmap=plt.cm.gray)
#     ax[3].set_title('{0} scale'.format(species.bin_nom))
#
#     plt.show()


def measure_all_leaves(all_species):
    for s in all_species:
        for p in s.get_leaf_paths():
            try:
                print('loading and measuring', s.bin_nom, 'at', p)
                r.load_new_image(p)
                # print("saving data to ruler")
                # r.save_data(s.bin_nom)
            except TypeError as e:
                print('Error encountered at leaf path "%0":'.format(p))
                print(e)


if __name__ == '__main__':
    r = Ruler()
    # use_ruler(Liriodendron_tulipifera, r)
    # print("saving data to ruler")
    # r.save_data(Liriodendron_tulipifera.bin_nom)
    s = load_all_species()
    measure_all_leaves(s)