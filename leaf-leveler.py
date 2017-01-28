import csv
from math import sqrt, acos, degrees
from os import listdir, walk
from os.path import isfile, join
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from skimage.morphology import binary_dilation, binary_erosion, skeletonize, remove_small_objects, binary_closing
from scipy.ndimage.interpolation import rotate
# from scipy import ndimage as ndi
# from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def level_leaf(array):
    """
    takes the extrema of the leaf, finds the
    hypotenuse, finds the angle between the base
    and the hypotenuse, rotates the image by that
    number, and reinitializes the leaf with the
    new image
    """

    edges_old = canny(array, 2.5)

    thresh = threshold_otsu(array)

    # binary = -(array > thresh)
    binary = array <= thresh
    binary = binary_erosion(binary_closing(binary_dilation(binary_dilation(binary))))

    edges = binary

    left_extrema = []  # stem
    right_extrema = []  # tip of leaf

    # Create an array of index values
    array_length = len(edges[0])
    left_to_right = range(0, array_length)
    for i in left_to_right:
        column = edges[:, i]
        # xvalue = np.argmax(self.edges[:, i])
        if column.any():
            left_extrema = [i, np.argmax(column)]
            break
    right_to_left = reversed(left_to_right)
    for i in right_to_left:
        column = edges[:, i]
        if column.any():
            right_extrema = [i, np.argmax(column)]
            break

    endpoints = [left_extrema, right_extrema]

    left_endpoint, right_endpoint = endpoints

    # find the distance (length) between the two points (hypotenuse)
    diff_x = right_endpoint[0] - left_endpoint[0]
    diff_y = right_endpoint[1] - left_endpoint[1]
    hypot = sqrt((diff_x) ** 2 + (diff_y) ** 2)

    # get the angle between the endpoints to rotate the image by
    angle_radians = acos(diff_x / hypot)
    angle_degrees = degrees(angle_radians)
    array = array.copy()

    # rotate the image, preserving size
    if diff_y < 0:
        array = rotate(array, -angle_degrees, reshape=True, mode='nearest')
    else:
        array = rotate(array, angle_degrees, reshape=True, mode='nearest')

    # reinitzialize the image again
    return array

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    while True:
        file_path = filedialog.askdirectory(
            initialdir='/home/patrick/PycharmProjects/SciFair-Y2/input-images/Year 1 Images'
        )

        for root, paths, filenames in walk(file_path):
            for f in filenames:
                if '.jpg' in f:
                    print(join(root, f))

                    leaf_img = imread(join(root, f), as_grey=True)
                    leveled = level_leaf(leaf_img)

                    leveled /= np.max(leveled)

                    imsave(join(root, f), leveled)