from os import getcwd
from os.path import join

import matplotlib.pyplot as plt

from main2 import load_species_structured
from ruler import Ruler, rotate_line_about_point

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
    sugar_maple = [species for species in leaves_list if species.bin_nom == 'Acer pensylvanicum'][0]

    leaf_path = sugar_maple.get_leaf_paths()[3]

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
    ax[0].set_title('{0}, {1}cm'.format(sugar_maple.bin_nom, length))
    row, col = img.shape
    #ax[1].axis((0, col, row, 0))
    ax[1].imshow(-img, cmap=plt.cm.gray)
    ax[1].set_title('Veins, unfiltered')
    center_line, above_line, below_line = None, None, None


    def plot_lines(lines, plot, color, label, linestyle):
        """
        A small function to plot all lines with regards to labeling.
        :param plot: the plot of matplotlib
        :param lines: a list of points to plot
        :param color: str of color
        :param label: str to name first segment
        :return: labeled line segment
        """

        first = True
        output_line = None
        for l in lines:
            if first:
                p0, p1 = l
                output_line, = plot.plot(
                    (p0[0], p1[0]),
                    (p0[1], p1[1]),
                    color=color,
                    marker='.',
                    markersize=5,
                    label=label,
                    linestyle=linestyle)
                first = False
            else:
                p0, p1 = l
                plot.plot(
                    (p0[0], p1[0]),
                    (p0[1], p1[1]),
                    color=color,
                    marker='.',
                    markersize=5,
                    linestyle=linestyle)
        first = True
        for l in lines:
            if first:
                p0, p1 = rotate_line_about_point(l, (500, 500), 0)
                output_line, = plot.plot(
                    (p0[0], p1[0]),
                    (p0[1], p1[1]),
                    color=color,
                    marker='.',
                    markersize=5,
                    label=label,
                    linestyle=linestyle)
                first = False
            else:
                p0, p1 = rotate_line_about_point(l, (500, 500), 0)
                plot.plot(
                    (p0[0], p1[0]),
                    (p0[1], p1[1]),
                    color=color,
                    marker='.',
                    markersize=5,
                    linestyle=linestyle)

    # colors for randomization
    colors = ['b', 'g', 'r', 'k', 'm', 'y', 'c', 'w']
    markers = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot', 'dotted']

    for count, cluster in enumerate(clustered_lines):
        num = count
        while num >= 8:
            num -= 8
        plot_lines(cluster,
                   ax[1],
                   color=colors[num],
                   label='Cluster {0}'.format(count+1),
                   linestyle=markers[num])

    # plot_lines(hough_center, ax[1], 'blue', 'Midrib veins')
    # plot_lines(hough_above, ax[1], 'green', 'Veins above midrib')
    # plot_lines(hough_below, ax[1], 'red', 'Veins below midrib')

    center_ceiling, = ax[1].plot((0, img.shape[1]), (hough_range[0], hough_range[0]), 'b--', label='Center ceiling')
    center_floor, = ax[1].plot((0, img.shape[1]), (hough_range[1], hough_range[1]), 'g--', label='Center floor')
    midrib_line, = ax[1].plot(endpoints[1], endpoints[0], label='Midrib approximation')

    plt.legend(
        # [
        #     above_line,
        #     center_line,
        #     below_line,
        #     center_ceiling,
        #     center_floor,
        #     midrib_line
        # ],
    )

    print(ruler.measure_surface_variability())

    plt.show()
