from math import atan, degrees, sqrt

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects, binary_closing
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import probabilistic_hough_line

# ~/.pyenv/versions/3.5.1/bin/python

"""
scale_reader.py: Reads raw images of leaves on the scale and converts those images
to measurements of the leaf.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas"]
__version__ = "1.0.0"
__date__ = "6/28/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

FLAT_LINE_SLOPE = 0.3
DEGREES_MARGIN = 10.0
CROP_MARGIN = -5  # number of pixels to add to prevent cutting the actual image


def import_leaf_image(path):
    """
    Load a leaf image on a scale and compute the leaf's features to the scale given
    :param path: path of the leaf image to be imported
    :return: measurement of leaf
    """
    original_image = imread(path, as_grey=True)[10:-10, 10:-10]
    scale, leaf = split_leaf_image(original_image)
    scale_cropped = auto_crop(scale)
    leaf_cropped = auto_crop(leaf)

    otsu = threshold_otsu(original_image)
    scale_binary = scale_cropped <= otsu
    leaf_binary = -remove_small_objects((leaf_cropped >= otsu))

    v_scale, h_scale = get_scale(scale_binary)

    # edges, lines, center_range = measure_veins(leaf_cropped, leaf_binary, v_scale, h_scale)
    edges, lines, lines2, lines3, center_y, center_range = measure_veins(leaf_cropped, leaf_binary, v_scale, h_scale)
    length = measure_leaf_length(leaf_binary, center_y, v_scale, h_scale)

    return leaf_cropped, lines, lines2, lines3, length, center_range


'''
    (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9, 4))[1]
    ax1.set_title('scale_cropped')
    ax1.imshow(scale_binary, cmap='gray')
    ax2.set_title('leaf_cropped')
    ax2.imshow(leaf_binary, cmap='gray')
    """
    ax2.plot((mid_start, mid_start),
             (0, original_image.shape[0]),
             'g-',
             linewidth=3)
    ax2.plot((mid_end, mid_end),
             (0, original_image.shape[0]),
             'r-',
             linewidth=3)
    """
    plt.show()
'''


def split_leaf_image(original_image):
    """
    Splits a leaf image based on the space between the scale and the leaf itself and returns the scale and leaf
    respectively.
    :param original_image: loaded numpy array of image
    :return: scale, leaf
    """
    otsu = threshold_otsu(original_image)
    binary = original_image <= otsu
    binary_clean = remove_small_objects(binary).astype(int)
    flattened_bin = []
    for column in binary_clean.T:
        flattened_bin.append(sum(column))
    # iterate over flattened_bin and look for 0 and non-zero values based on conditions of already found features
    scale_found = False
    scale_start = None
    space_found = False
    mid_start = None
    mid_end = None
    for count, c_sum in enumerate(flattened_bin):
        if not scale_found and c_sum > 0:
            scale_found = True
            scale_start = count
        elif scale_found:
            if c_sum == 0 and scale_found and not space_found:
                space_found = True
                mid_start = count
            elif c_sum > 0 and space_found:
                mid_end = count
                break
    mid_mid = int((mid_start + mid_end) / 2)
    scale = original_image[:, 0:mid_mid]
    leaf = original_image[:, mid_mid:]
    return scale, leaf


def auto_crop(image):
    """
    Takes "image" and automatically crops it like the GIMP autocrop feature.
    Iterates over each row and column of the image unitl it finds the object, then it crops the image as close as
    possible.
    :param image: normally either scale or leaf
    :return: cropped_image
    """
    otsu = threshold_otsu(image)
    binary = image <= otsu
    binary_clean = remove_small_objects(binary).astype(int)
    h_crop = [None, None]
    v_crop = [None, None]
    # From the top of the image
    for count, row in enumerate(binary_clean):
        if sum(row) > 0:
            v_crop[1] = count - CROP_MARGIN
    # From the bottom of the image (top flipped h)
    for count, row in enumerate(binary_clean[::-1, :]):
        if sum(row) > 0:
            v_crop[0] = image.shape[0] - count + CROP_MARGIN
    # From the left of the image (top 90 degrees clockwise)
    for count, row in enumerate(binary_clean.T):
        if sum(row) > 0:
            h_crop[1] = count - CROP_MARGIN
    # From the right of the image (top 90 degrees clockwise flipped h)
    for count, row in enumerate(binary_clean.T[::-1, :]):
        if sum(row) > 0:
            h_crop[0] = image.shape[1] - count + CROP_MARGIN
    return image[v_crop[0]:v_crop[1], h_crop[0]:h_crop[1]]


def get_scale(scale_binary):
    """
    Takes the binary scale and measures the vertical and horizontal scale based on the tips of the diamonds.
    Each "scale length" is each to 2 cm
    :param scale_binary:
    :return:
    """
    v_pos1 = None
    v_pos2 = None
    h_pos1 = None
    h_pos2 = None

    # vertical
    # top
    for r_num, row in enumerate(scale_binary[::, ::]):
        for c_num, column in enumerate(row):
            if column:
                v_pos1 = (c_num, r_num)
                break
    # bottom
    for r_num, row in enumerate(scale_binary[::-1, ::]):
        for c_num, column in enumerate(row):
            if column:
                v_pos2 = (c_num, scale_binary.shape[1] - r_num)
                break
    # horizontal
    # left
    for c_num, column in enumerate(scale_binary.T[::, ::]):
        for r_num, row in enumerate(column):
            if row:
                h_pos1 = (c_num, r_num)
                break
    # right
    for c_num, column in enumerate(scale_binary.T[::-1, ::]):
        for r_num, row in enumerate(column):
            if row:
                h_pos2 = (scale_binary.shape[0] - c_num, r_num)
                break

    v_cm = dist_form(v_pos1, v_pos2) / 2.0
    h_cm = dist_form(h_pos1, h_pos2) / 2.0
    return v_cm, h_cm


def dist_form(coord1, coord2, h_scale=1, v_scale=1):
    x = ((coord1[0] - coord2[0]) / float(h_scale)) ** 2
    y = ((coord1[1] - coord2[1]) / float(v_scale)) ** 2
    dist = sqrt(x + y)
    return dist


def line_to_slope(coord1, coord2):
    """
    A simple function to get the slope between two lines
    :param coord1:
    :param coord2:
    :return:
    """
    try:
        slope = float(coord1[1] - coord2[1]) / float(coord1[0] - coord2[0])
        return slope
    except ZeroDivisionError:
        return 0.


def line_to_angle(line):
    """
    Takes a line and uses arctangent and degrees() to find the angle of a line
    :param line: (point1, point2)
    :return: angle in degrees
    """
    p0, p1 = line
    y_diff = p0[1] - p1[1]
    x_diff = p0[0] - p1[0]
    try:
        return degrees(atan(y_diff / x_diff))
    except ZeroDivisionError:
        return 90


def measure_leaf_length(leaf_binary, center_y, v_scale, h_scale):
    """
    Measures the leaf in terms of the scale
    :param leaf_binary: binary representation of the leaf
    :param v_scale: vertical scale for 2 cm
    :param h_scale: horizontal scale for 2 cm
    :return: length in cm
    """
    # # find the endpoints
    # for count, pixel in enumerate(leaf_binary.T[0, :]):
    #     if pixel:
    #         endpoint1 = [0, count]
    #         break
    # for count, pixel in enumerate(leaf_binary.T[-1, :]):
    #     if pixel:
    #         endpoint2 = [leaf_binary.shape[1], count]
    #         break
    # length = dist_form(endpoint1, endpoint2, h_cm, v_cm)
    # print('{0} cm'.format(str(length)))
    # return length
    v_cm = v_scale / 2.0
    h_cm = h_scale / 2.0
    endpoint1 = None
    endpoint2 = None
    slice = leaf_binary[center_y]
    p0 = None
    p1 = None
    for count, p in enumerate(slice):
        if p:
            p0 = count
            break
    for count, p in enumerate(slice[::-1]):
        if p:
            p1 = slice.shape[0] - count
            break
    return float(p1 - p0) / h_cm


def measure_veins(leaf_img, leaf_binary, v_scale, h_scale):
    """
    Extract and measure the veins in leaf_img
    :param leaf_img: grayscale img of leaf (cropped)
    :param leaf_binary: binary leaf
    :param v_scale: vertical scale
    :param h_scale: horizontal scale
    :return:
    """
    equalized = equalize_adapthist(leaf_img, clip_limit=0.03)
    denoised = denoise_tv_chambolle(equalized, weight=0.2, multichannel=True)

    leaf_bitmap = denoised.copy()

    # only show leaf that is not background with threshold
    for row_count, row in enumerate(binary_dilation(leaf_binary.astype(bool))):
        for column_count, pixel in enumerate(row):
            if not pixel:
                leaf_bitmap[row_count, column_count] = 1

    edges = canny(leaf_bitmap, sigma=2.5)
    edges = binary_closing(edges)
    vein_edges = edges - np.logical_and(edges, -binary_erosion(leaf_binary))
    lines = probabilistic_hough_line(vein_edges, threshold=10, line_length=50, line_gap=10)

    level_lines = []
    for l in lines:
        if -FLAT_LINE_SLOPE < line_to_slope(l[0], l[1]) < FLAT_LINE_SLOPE:
            level_lines.append(l)

    # get the median level line that will hopefully be the midvein.
    rough_y_center = np.median([np.average([p0[1], p1[1]]) for p0, p1 in level_lines])
    # calculate the range in which a line will be considered part of the midvein
    center_range = [
        rough_y_center - leaf_binary.shape[1] * 0.025,
        rough_y_center + leaf_binary.shape[1] * 0.025
    ]
    # collect all lines completely within the range
    center_lines = []
    for p0, p1 in level_lines:
        if center_range[0] < p0[1] < center_range[1] and center_range[0] < p1[1] < center_range[1]:
            center_lines.append((p0, p1))
    center_y = np.average([[p0[1], p1[1]] for p0, p1 in center_lines])
    # center_slope = np.average([get_slope(p0, p1) for p0, p1 in center_lines])
    center_line = ((0, center_y), (leaf_binary.shape[0], center_y))

    center_lines.append(center_line)

    # code dealing with nonflat lines
    # separate lines based on above and below center line
    above = []
    below = []
    for l in lines:
        p0, p1 = l
        if l not in level_lines:
            if center_y <= p0[1] and center_y <= p1[1]:
                below.append([l, line_to_slope(p0, p1), line_to_angle(l)])
            elif center_y >= p0[1] and center_y >= p1[1]:
                above.append([l, line_to_slope(p0, p1), line_to_angle(l)])

    # ABOVE
    above_filtered_1 = []
    above_angles = []
    for pair in above:
        l, slope, angle = pair
        # remove lines that are above the center that have a negative slope
        if slope < 0.:
            above_angles.append(angle)
            above_filtered_1.append(pair)

    # remove all lines that are not within the margin of degrees of the median line
    above_filtered_2 = []
    margin = [np.median(above_angles) - DEGREES_MARGIN,
              np.median(above_angles) + DEGREES_MARGIN]
    for l, slope, angle in above_filtered_1:
        if margin[0] < angle < margin[1]:
            above_filtered_2.append(l)

    # BELOW
    below_filtered_1 = []
    below_angles = []
    for pair in below:
        l, slope, angle = pair
        # remove lines that are below the center that have a negative slope
        if slope > 0.:
            below_angles.append(angle)
            below_filtered_1.append(pair)

    # remove all lines that are not within the margin of degrees of the median line
    below_filtered_2 = []
    margin = [np.median(below_angles) - DEGREES_MARGIN,
              np.median(below_angles) + DEGREES_MARGIN]
    for l, slope, angle in below_filtered_1:
        if margin[0] < angle < margin[1]:
            below_filtered_2.append(l)

    return vein_edges, above_filtered_2, below_filtered_2, center_lines, center_y, center_range


class Ruler():
    """
    Provides a class to temporarily store and measure leaves easily.

    New version of the scale_reader functions (essentially all scale_reader.py functions
    in a class.
    """

    def __init__(self):
        self.current_path = None
        self.img = None
        self.scale = None
        self.leaf = None
        self.otsu = None
        self.scale_bin = None
        self.leaf_bin = None
        self.v_cm = None
        self.h_cm = None

        self.vein_measure = None
        self.length = None

    def load_new_image(self, path):
        self.__init__()

        self.current_path = path

        self.img = imread(path, as_grey=True)[10:-10, 10:-10]

        self.otsu = threshold_otsu(self.img)
        self.split_leaf_image()
        self.scale = auto_crop(self.scale)
        self.leaf = auto_crop(self.leaf)
        self.scale_bin = self.scale <= self.otsu
        self.leaf_bin = -remove_small_objects((self.leaf >= self.otsu))
        self.v_cm, self.h_cm = self.get_scale()

        self.vein_measure = self.measure_veins()
        self.length = self.measure_length()

    def split_leaf_image(self):
        binary = self.img <= self.otsu
        binary_clean = remove_small_objects(binary,
                                            min_size=64,
                                            connectivity=1,
                                            in_place=False).astype(int)
        flat = []
        for column in binary_clean.T:
            flat.append(sum(column))
        # iterate over flat and look for 0 and non-zero values based on
        # conditions of already found features
        scale_found = False
        space_found = False
        mid_start = None
        mid_end = None
        for count, c_sum in enumerate(flat):
            if not scale_found and c_sum > 0:
                scale_found = True
            elif scale_found:
                if c_sum == 0 and scale_found and not space_found:
                    space_found = True
                    mid_start = count
                elif c_sum > 0 and space_found:
                    mid_end = count
                    break
        center_split = int((mid_start + mid_end) / 2)
        scale = self.img[:, 0:center_split]
        leaf = self.img[:, center_split:]
        return scale, leaf

    def auto_crop(self, image):
        binary = image <= self.otsu
        binary_clean = remove_small_objects(binary).astype(int)
        h_crop = [None, None]
        v_crop = [None, None]
        # From the top of the image
        for count, row in enumerate(binary_clean):
            if sum(row) > 0:
                v_crop[1] = count - CROP_MARGIN
        # From the bottom of the image (top flipped h)
        for count, row in enumerate(binary_clean[::-1, :]):
            if sum(row) > 0:
                v_crop[0] = image.shape[0] - count + CROP_MARGIN
        # From the left of the image (top 90 degrees clockwise)
        for count, row in enumerate(binary_clean.T):
            if sum(row) > 0:
                h_crop[1] = count - CROP_MARGIN
        # From the right of the image (top 90 degrees clockwise flipped h)
        for count, row in enumerate(binary_clean.T[::-1, :]):
            if sum(row) > 0:
                h_crop[0] = image.shape[1] - count + CROP_MARGIN
        return image[v_crop[0]:v_crop[1], h_crop[0]:h_crop[1]]

    def get_scale(self):
        v_pos1 = None
        v_pos2 = None
        h_pos1 = None
        h_pos2 = None

        # vertical
        # top
        for r_num, row in enumerate(self.scale_bin[::, ::]):
            for c_num, column in enumerate(row):
                if column:
                    v_pos1 = (c_num, r_num)
                    break
        # bottom
        for r_num, row in enumerate(self.scale_bin[::-1, ::]):
            for c_num, column in enumerate(row):
                if column:
                    v_pos2 = (c_num, self.scale_bin.shape[1] - r_num)
                    break
        # horizontal
        # left
        for c_num, column in enumerate(self.scale_bin.T[::, ::]):
            for r_num, row in enumerate(column):
                if row:
                    h_pos1 = (c_num, r_num)
                    break
        # right
        for c_num, column in enumerate(self.scale_bin.T[::-1, ::]):
            for r_num, row in enumerate(column):
                if row:
                    h_pos2 = (self.scale_bin.shape[0] - c_num, r_num)
                    break

        v_cm = dist_form(v_pos1, v_pos2) / 2.0
        h_cm = dist_form(h_pos1, h_pos2) / 2.0
        return v_cm, h_cm

    def measure_veins(self):
        equalized = equalize_adapthist(self.leaf, clip_limit=0.03)
        denoised = denoise_tv_chambolle(equalized, weight=0.2, multichannel=True)

        leaf_bitmap = denoised.copy()

        # only show leaf that is not background with threshold
        for row_count, row in enumerate(binary_dilation(self.leaf_bin.astype(bool))):
            for column_count, pixel in enumerate(row):
                if not pixel:
                    leaf_bitmap[row_count, column_count] = 1

        edges = canny(leaf_bitmap, sigma=2.5)
        edges = binary_closing(edges)
        vein_edges = edges - np.logical_and(edges, -binary_erosion(self.leaf_bin))
        lines = probabilistic_hough_line(vein_edges, threshold=10, line_length=50, line_gap=10)

        level_lines = []
        for l in lines:
            if -FLAT_LINE_SLOPE < line_to_slope(l[0], l[1]) < FLAT_LINE_SLOPE:
                level_lines.append(l)

        # get the median level line that will hopefully be the midvein.
        rough_y_center = np.median([np.average([p0[1], p1[1]]) for p0, p1 in level_lines])
        # calculate the range in which a line will be considered part of the midvein
        center_range = [
            rough_y_center - self.leaf_bin.shape[1] * 0.025,
            rough_y_center + self.leaf_bin.shape[1] * 0.025
        ]
        # collect all lines completely within the range
        center_lines = []
        for p0, p1 in level_lines:
            if center_range[0] < p0[1] < center_range[1] and center_range[0] < p1[1] < center_range[1]:
                center_lines.append((p0, p1))
        center_y = np.average([[p0[1], p1[1]] for p0, p1 in center_lines])
        # center_slope = np.average([get_slope(p0, p1) for p0, p1 in center_lines])
        center_line = ((0, center_y), (self.leaf_bin.shape[0], center_y))

        center_lines.append(center_line)

        # code dealing with nonflat lines
        # separate lines based on above and below center line
        above = []
        below = []
        for l in lines:
            p0, p1 = l
            if l not in level_lines:
                if center_y <= p0[1] and center_y <= p1[1]:
                    below.append([l, line_to_slope(p0, p1), line_to_angle(l)])
                elif center_y >= p0[1] and center_y >= p1[1]:
                    above.append([l, line_to_slope(p0, p1), line_to_angle(l)])

        # ABOVE
        above_filtered_1 = []
        above_angles = []
        for pair in above:
            l, slope, angle = pair
            # remove lines that are above the center that have a negative slope
            if slope < 0.:
                above_angles.append(angle)
                above_filtered_1.append(pair)

        # remove all lines that are not within the margin of degrees of the median line
        above_filtered_2 = []
        margin = [np.median(above_angles) - DEGREES_MARGIN,
                  np.median(above_angles) + DEGREES_MARGIN]
        for l, slope, angle in above_filtered_1:
            if margin[0] < angle < margin[1]:
                above_filtered_2.append(l)

        # BELOW
        below_filtered_1 = []
        below_angles = []
        for pair in below:
            l, slope, angle = pair
            # remove lines that are below the center that have a negative slope
            if slope > 0.:
                below_angles.append(angle)
                below_filtered_1.append(pair)

        # remove all lines that are not within the margin of degrees of the median line
        below_filtered_2 = []
        margin = [np.median(below_angles) - DEGREES_MARGIN,
                  np.median(below_angles) + DEGREES_MARGIN]
        for l, slope, angle in below_filtered_1:
            if margin[0] < angle < margin[1]:
                below_filtered_2.append(l)

        return {'canny edges': vein_edges,
                'hough above': above_filtered_2,
                'hough below': below_filtered_2,
                'hough center': center_lines,
                'midvein': center_y,
                'center range': center_range}

    def measure_length(self):
        y = self.vein_measure['midvein']
        p0 = None
        p1 = None
        for count, pixel in enumerate(self.leaf_bin[y, ::]):
            if pixel:
                p0 = (count, y)
                break
        for count, pixel in enumerate(self.leaf_bin[y, ::-1]):
            if pixel:
                p1 = (self.leaf_bin.shape[0] - count, y)
                break
        length = dist_form(p0, p1, v_scale=self.v_cm, h_scale=self.h_cm)
        return length
