from os import getcwd
from os.path import join
from random import randint

import matplotlib.pyplot as plt

from main2 import load_species_structured
from ruler import Ruler

# ~/.pyenv/versions/3.5.2/bin/python

"""
clustering_test.py: A script to test the ruler class and ensure functionality in regards to clustering
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "11/30/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

# global variables
IMAGE_DIR = join(getcwd(), 'input-images')  # master image directory

# start main
if __name__ == '__main__':
    # create ruler
    ruler = Ruler()

    # use_ruler(Liriodendron_tulipifera, r)
    # print("saving data to ruler")
    # r.save_data(Liriodendron_tulipifera.bin_nom)

    # load the leaves from the harddrive
    leaves_list = load_species_structured(IMAGE_DIR)

    # get a leaf
    selected_leaf = [species for species in leaves_list if species.bin_nom == 'Acer pensylvanicum'][0]

    leaf_path = selected_leaf.get_leaf_paths()[3]

    # load the leaf with the ruler
    ruler.load_new_image(leaf_path)
    print(str(ruler))

    # img, lines, lines2, lines3, length, center_range
    img = ruler.leaf
    hough_center = ruler.vein_measure['hough center']
    hough_above = ruler.vein_measure['hough above']
    hough_below = ruler.vein_measure['hough below']
    hough_range = ruler.vein_measure['center range']
    midrib_line = ruler.vein_measure['midrib lin approx']
    length = ruler.length

    clustered_lines = ruler.__measure_veins_group_veins__(hough_above)

    ep0, ep1 = ruler.endpoints

    endpoints = ((ep0[1], ep1[1]), (ep0[0], ep1[0]))

    # displaying data with pyplot and matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    ax = axes.ravel()
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('{0}, {1}cm'.format(selected_leaf.bin_nom, length))
    row, col = img.shape
    ax[1].axis((0, col, row, 0))
    ax[1].imshow(-img, cmap=plt.cm.gray)
    ax[1].set_title('Veins, unfiltered')
    center_line, above_line, below_line = None, None, None


    def plot_lines(line_list, plot, color, label, linestyle):
        """
        A small function to plot all lines with regards to labeling.
        :param plot: the plot of matplotlib
        :param line_list: a list of points to plot
        :param color: str of color
        :param label: str to name first segment
        :param linestyle: style of lines, str
        :return: labeled line segment
        """

        first = True
        output_line = None
        for l in line_list:
            if first:
                line_p0, line_p1 = l
                output_line, = plot.plot(
                    (line_p0[0], line_p1[0]),
                    (line_p0[1], line_p1[1]),
                    color=color,
                    marker='.',
                    markersize=10,
                    label=label,
                    linestyle=linestyle)
                first = False
            else:
                line_p0, line_p1 = l
                plot.plot(
                    (line_p0[0], line_p1[0]),
                    (line_p0[1], line_p1[1]),
                    color=color,
                    marker='.',
                    markersize=10,
                    linestyle=linestyle)

        return output_line

    # colors for randomization
    colors = ['b', 'g', 'r', 'k', 'm', 'y', 'c', 'w']
    markers = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot', 'dotted']

    contours_dict, angles_dict = ruler.measure_contours(levels=3)
    all_keys = [k for k in contours_dict.keys()]
    all_keys.sort()

    lin_approx = ruler.__measure_veins_create_midrib_lin_approx__()

    lines = []

    for i in range(0, 7):
        p0, p1 = contours_dict[i]
        lines.append((p0, p1))

    plot_lines(lines,
               ax[1],
               color=colors[3],
               label='3 Levels, 8 Sections',
               linestyle=markers[2])

    # display legend
    plt.legend()

    # display plot
    plt.show()
