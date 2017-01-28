import xml.etree.cElementTree as ETree
from math import tan, radians
from os.path import join, isfile
from warnings import warn
from math import sin, cos, pi, sqrt

import sys
import traceback

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.io import imread, imshow, show
from skimage.measure import perimeter
from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects, binary_closing
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import probabilistic_hough_line

from scale_reader import dist_form, line_to_angle, line_to_slope

# ~/.pyenv/versions/3.5.2/bin/python

"""
ruler.py: Provides a class to measure and temporarily store leaves and their
measurements.

Uses many of the functions from scale_reader.py, but those functions are reorganized
into a class that is better suited for batch operations.

Hopefully the the way this is designed is that it will reduce memory usage by
constantly refreshing the class's variables with each new leaf image to be measured.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "7/8/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

DEGREES_MARGIN = 15.0  # margin of degrees to determine what is a similar hough line
# CROP_MARGIN = -5  # number of pixels to add to prevent cutting the actual image
CROP_MARGIN = 0

ADAPTHIST_CLIP_LIMIT = 0.03
DENOISE_WEIGHT = 0.2
CANNY_SIGMA = 2.5
HOUGH_PARAMS = (10, 50, 10)

MIN_TOTAL_LINES = 18
MIN_OTHER_LINES = 6
FLAT_LINE_DEGREES = 15.0  # slope range used  to determine flat lines (essentially +- #)
MIDRIB_MARGIN_PERCENT = 0.05

CONTOUR_LEVELS = 3

# default scale
DEFAULT_SCALE = imread('/home/patrick/PycharmProjects/SciFair-Y2/input-images/default_scale.png', as_grey=True)


class Ruler:
    """
    Provides a class to temporarily store and measure leaves easily.

    New version of the scale_reader functions (essentially all scale_reader.py functions
    in a class.
    """

    def __init__(self):
        """
        Initializes all variables to none.

        Designed to be used repeatedly for many leaves.
        """

        # the 'cwd' of the leaf
        self.current_path = None

        # raw image
        self.img = None

        # separated leaf and scale, in grayscale
        self.scale = None
        self.leaf = None

        # Ostu value
        self.otsu = None

        # binary separated images of leaf and scale
        self.scale_bin = None
        self.leaf_bin = None

        # scale of leaf based off of visual scale
        self.vertical_cm_scale = None
        self.horizontal_cm_scale = None

        # vein specific measurements
        '''
        self.vein_measure contains the following:
        {
            'canny edges':  vein_edges,
            'hough above':  above,
            'hough below':  below,
            'hough center': midrib_lines,
            'midvein':      center_y,
            'center range': midrib_range,
            'midrib line':  midrib_lin_approx,
        }
        '''
        self.vein_measure = None
        self.length = None
        self.midrib_func = None
        self.endpoints = None

        self.area = None

        # perimeter
        self.perimeter_v = None
        self.perimeter_h = None
        self.perimeter_p = None

        # surface variability
        self.surf_var_mean = None
        self.surf_var_median = None

        # contour and angle data
        self.contours_pos = None
        self.contours_size = None
        self.contours_angles = None

        # dict o data
        self.data_dict = None

    def load_new_image(self, path, no_scale=False):
        """
        Loads a leaf image and measures the features of the leaf.
        Also updates the varaibles of the ruler class to reflect the currently measured
        leaf.
        :param no_scale: whether or not to look of a scale in the image provided. useful for year 1 leaves.
        :param path: path of a leaf image
        :return: None
        """
        # reinit all values to None
        self.__init__()

        # set the cwd to the leaf's own path
        self.current_path = path

        # load the leaf from the drive as a grayscale image
        # also slightly crop the image to remove any 'framing' of the picture
        self.img = imread(path, as_grey=True)[5:-5, 5:-5]

        # obtain an Otsu's value
        self.otsu = threshold_otsu(self.img)

        # if scale (default)
        if not no_scale:
            # split the image of the leaf automatically
            self.scale, self.leaf = self.split_leaf_image()

            # reduce whitespace around leaf and scale
            self.scale = self.auto_crop(self.scale)
            self.leaf = self.auto_crop(self.leaf)

            # reduce both parts to binary and remove artifacts from image
            self.scale_bin = np.bool_(self.scale <= self.otsu)
            self.leaf_bin = -remove_small_objects(np.bool_(self.leaf >= self.otsu))

        # no scale mode for legacy leaf pictures
        elif no_scale:
            # split the image of the leaf automatically
            self.scale = DEFAULT_SCALE
            self.leaf = self.img

            # only crop the leaf
            self.leaf = self.auto_crop(self.leaf)

            # reduce both parts to binary and remove artifacts from image
            self.scale_bin = self.scale.astype(bool)
            self.leaf_bin = -remove_small_objects(np.bool_(self.leaf >= self.otsu))

        # measure the scale from the binary scale image
        self.vertical_cm_scale, self.horizontal_cm_scale = self.get_scale()

        # measure vein data
        self.vein_measure = self.measure_veins()

        # measure length and perimeter
        self.length = self.measure_length()
        self.perimeter_p, self.perimeter_h, self.perimeter_v = self.measure_perimeter()

        # measure area directly
        self.area = np.sum(self.leaf_bin) / (self.vertical_cm_scale * self.horizontal_cm_scale)

        # surface variability
        self.surf_var_mean, self.surf_var_median = self.measure_surface_variability()

        # contours
        self.contours_pos, self.contours_size, self.contours_angles = self.measure_contours(levels=CONTOUR_LEVELS)

        # for containing all values for xml saving
        self.data_dict = dict(path=str(self.current_path),
                              v_cm=str(self.vertical_cm_scale),
                              h_cm=str(self.horizontal_cm_scale),
                              otsu=str(self.otsu),
                              p=str(self.perimeter_p),

                              length=str(self.length),
                              p_v=str(self.perimeter_h),
                              p_h=str(self.perimeter_v),
                              area=str(self.area),
                              sf_v_mean=str(self.surf_var_mean),
                              sf_v_median=str(self.surf_var_median),

                              array_files='unknown')

    def __str__(self):
        """
        Returns a useful readout of the current leaf's measurements.
        :return: string of measurements
        """
        string = """
MEASUREMENTS OF CURRENT LEAF

PATH                    {0}

CM SCALE    VERTICAL    {1}
            HORIZONTAL  {2}

OTSU                    {3}

PERIMETER   CENTIMETERS {4}

LENGTH      CENTIMETERS {5}

ARRAY FILES PATH        {6}""".format(
            self.data_dict['path'],
            self.data_dict['v_cm'],
            self.data_dict['h_cm'],
            self.data_dict['otsu'],
            self.data_dict['p'],
            self.data_dict['length'],
            self.data_dict['array_files'],
        )

        return string

    def generate_data_dict(self, bin_nom, save_data_path='save-data'):
        """
        Generates a dictionary of data that contains the most useful data from the leaf.

        :param bin_nom: the name of the leaf (str)
        :param save_data_path: the save path of the leaf (str)
        :return: dict of data
        """

        # get the file's name
        img_filename = self.current_path.split('/')[-1]  # last part of dir. split by slashes

        # generate file name based on img and species
        array_save_data_filename = "{0} - {1}".format(bin_nom, img_filename.split('.')[0])
        array_save_data_path = join(save_data_path, array_save_data_filename)

        data = {
            # data that isn't all that important to the ANN
            'path':             str(self.current_path),
            'v_cm':             str(self.vertical_cm_scale),
            'h_cm':             str(self.horizontal_cm_scale),
            'otsu':             str(self.otsu),

            # simple measurements of a leaf (one value)
            'p':                str(self.perimeter_p / ((self.perimeter_h + self.perimeter_v) / 2)),
            'length':           str(self.length),
            'area':             str(self.area),
            'sf_v_mean':        str(self.surf_var_mean),
            'sf_v_median':      str(self.surf_var_median),

            # vein measurements
            'vein_angle_above': str(self.vein_measure['angle above']),
            'vein_angle_below': str(self.vein_measure['angle below']),
            'vein_length':      str(self.vein_measure['vein length']),

            # save location
            'array_files':      array_save_data_path + '.npz'
        }

        return data

    # save the currently loaded leaf, appended to the current .xml file
    def save_data(self, bin_nom, save_data_path='save-data', xml_filename='leaf-data.xml'):
        """
        Save the currently loaded leaf to the .xml file given by fname. The leaf is saved
        under the species bin_nom.
        :param bin_nom: name of leave species
        :param save_data_path: path of save_data
        :param xml_filename: name of the .xml file
        :return: None
        """

        # create new fname if fname doesn't exist on disk
        if not isfile(join(save_data_path, xml_filename)):
            # create preliminary 'data' tree
            data = ETree.Element('data')

            # old code to create a testspecies
            # species = ETree.SubElement(data, 'species')
            # ETree.SubElement(species, 'g').text = "testgenus"
            # ETree.SubElement(species, 's').text = "testspecies"

            # make a xml tree
            tree = ETree.ElementTree(data)

            # write tree to disk
            tree.write(join(save_data_path, "leaf-data.xml"))

        # load tree, should exist given previous if-statement
        tree = ETree.parse(join(save_data_path, xml_filename))

        # get the root of the file
        root = tree.getroot()

        # initalize values
        child = None  # tree of matching species
        species_found = False  # whether species exists in xml file or not

        # for all species listed under <data>
        for species in root.findall('species'):
            # get the bin. nom. of the species
            g = species.find('g')
            s = species.find('s')

            # if the species' actually exists (and is not empty)
            if g is not None and s is not None:
                # and if the name of the species in the file matches the current leaf's bin_nom
                if '{0} {1}'.format(g.text, s.text) == bin_nom:
                    # matching species is found
                    species_found = True
                    child = species
                    break

        # Make a new species if the species is not in the XML file
        if not species_found:
            # make own child
            child = ETree.SubElement(root, 'species')

            # split bin_nom to create species with specified data
            g_str, s_str = bin_nom.split()

            # add the text to the species (child tree)
            g = ETree.SubElement(child, 'g')
            s = ETree.SubElement(child, 's')
            g.text = g_str
            s.text = s_str

        # get the file's name
        img_filename = self.current_path.split('/')[-1]  # last part of dir. split by slashes

        # generate file name based on img and species
        array_save_data_filename = "{0} - {1}".format(bin_nom, img_filename.split('.')[0])
        array_save_data_path = join(save_data_path, array_save_data_filename)

        # # determine if specific leaf is already in xml file (CURRENTLY UNNEEDED)
        # img_found = False

        # find all specific leaves in species category
        for l in child.findall('leaf'):
            # if a saved leaf's img name matches this leaf's img name
            if l.attrib['name'] == img_filename:
                # warn that this leaf already exists
                warn_msg = "Leaf '{0}' already exists in {1}".format(img_filename, xml_filename)
                warn(warn_msg, Warning)

                # # say that image has been found
                # img_found = True

                # remove leaf's entry
                child.remove(l)

                break

        # create new subelement of species by the leaf
        leaf = ETree.SubElement(child, 'leaf', attrib={'name': img_filename})

        # Write data to leaf's entry
        self.data_dict = self.generate_data_dict(bin_nom, save_data_path=save_data_path)

        # # more for organization
        # arrays = [
        #     self.img,
        #     self.scale,
        #     self.leaf,
        #     self.scale_bin,
        #     self.leaf_bin,
        #     self.vein_measure,
        #     self.midrib_func,
        #     self.endpoints,
        # ]

        # write all xml compatible values in the dict as well as paths to arrays
        for attribute in self.data_dict.keys():
            # set all of the leaf's attributes in the xml tree to what they are supposed to be
            ETree.SubElement(leaf, attribute).text = self.data_dict[attribute]

        # save xml to file
        tree.write(join(save_data_path, xml_filename))

        # save all arrays (data to large to save in xml file and uncommonly used) into compressed .npz file

        # remove the midrib linear approximation since it cannot be saved
        self.vein_measure.pop('midrib lin approx', None)

        # save arrays as compressed numpy format
        np.savez_compressed(array_save_data_path,
                            img=self.img,
                            scale=self.scale,
                            leaf=self.leaf,
                            scale_bin=self.scale_bin,
                            leaf_bin=self.leaf_bin,
                            veins=self.vein_measure,
                            midrib=self.midrib_func,
                            endpoints=self.endpoints,
                            contour_pos=self.contours_pos,
                            contour_size=self.contours_size,
                            contour_angles=self.contours_angles,
                            )

        return None

    '''
    def load_from_xml(self):
        """
        Load all values of a given leaf from the xml file (and corresponding array file).
        :return: None
        """
        return None
    '''

    # chapter 1: preparing the images to be measured
    # step 0: basically initializing the images to be measured

    def split_leaf_image(self):
        """
        Splits self.img by the whitespace between the scale and the leaf itself.
        Separates the scale and the leaf.
        :return: scale, leaf
        """

        # reduces image to binary form based on Otsu value
        binary = self.img <= self.otsu

        # remove all small artifacts on the image (real image of this method not completely known
        binary_clean = remove_small_objects(binary,
                                            min_size=64,
                                            connectivity=1,
                                            in_place=False).astype(int)

        # 'flatten' the image to a list of the sum of the columns
        # useful for detecting presence of leaf and scale on img
        # should be 0 when there is no leaf nor scale
        flat = []
        for column in binary_clean.T:
            flat.append(sum(column))

        # iterate over flat and look for 0 and non-zero values based on
        # conditions of already found features

        # boolean values to mark beginning and end of searches for certain features
        scale_found = False
        space_found = False

        # actual range of mid section
        mid_start = None
        mid_end = None

        # iterate through image
        for count, c_sum in enumerate(flat):
            # catch when scale not found and something in image
            if not scale_found and c_sum > 0:
                scale_found = True

            # then catch when scale found and
            elif scale_found:
                # space between scale and leaf not found
                if c_sum == 0 and scale_found and not space_found:
                    space_found = True
                    mid_start = count

                # space is found and nothing in image (end of leaf)
                elif c_sum > 0 and space_found:
                    mid_end = count - 1
                    break

        # split the image between the bounds of the middle space
        center_split = int((mid_start + mid_end) / 2)

        # split image based on center value
        scale = self.img[:, 0:center_split]
        leaf = self.img[:, center_split:]

        return scale, leaf

    # automatically closely crop the image
    def auto_crop(self, image):
        """
        Automatically crops the leaf to within CROP_MARGIN of the closest pixels of
        the thresholded image by the thresholded image.
        :param image: Image to be automatically cropped.
        :return: Cropped image
        """

        # threshold image by Otsu number
        bin_img = image <= self.otsu

        # convert to binary as none ofthe function need the image ina non-binary format.
        bin_img = bin_img.astype(bool)

        # Remove holes in the leaf, but increases the size of rouge dots that are not the leaf.
        bin_img = binary_closing(binary_dilation(bin_img))
        bin_img = binary_dilation(binary_closing(bin_img))

        # remove "islands" in the picture. Time consuming but effective in removing any erraneous dots.
        bin_img = remove_small_objects(bin_img, min_size=100, connectivity=2)

        # variables to store the number of pixels to cut the leaf by
        h_crop = [None, None]
        v_crop = [None, None]

        # get the number of pixels to remove off of each side of the image.
        # done by getting a sum of a row or column and testing if the row or column has nothing in

        # each if-statement iterates through the image, however each if-statement transforms the image to get the
        # desired crop margin

        # it (sum of 0)

        # From the top of the image
        for count, row in enumerate(bin_img):
            if np.any(row):
                v_crop[1] = count - CROP_MARGIN

        # From the bottom of the image (top flipped h)
        for count, row in enumerate(bin_img[::-1, :]):
            if np.any(row):
                v_crop[0] = image.shape[0] - count + CROP_MARGIN

        # From the left of the image (top 90 degrees clockwise)
        for count, row in enumerate(bin_img.T):
            if np.any(row):
                h_crop[1] = count - CROP_MARGIN

        # From the right of the image (top 90 degrees clockwise flipped h)
        for count, row in enumerate(bin_img.T[::-1, :]):
            if np.any(row):
                h_crop[0] = image.shape[1] - count + CROP_MARGIN

        return image[v_crop[0]:v_crop[1], h_crop[0]:h_crop[1]].copy()

    # step 1: obtain scale

    def get_scale(self):
        """
        Measures the scale of the leaf image by iterating through the array
        until pixels of the scale are found. Once points are found, the dist_form
        function from scale_reader.py returns the scale.
        :return: v_cm and h_cm, the vertical and horizontal measures of a centimeter
        """

        # initialize values (for corners of scale's square)
        v_pos1 = None
        v_pos2 = None
        h_pos1 = None
        h_pos2 = None

        # vertical
        #
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
        #
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

        # calculate distance between points
        v_cm = dist_form(v_pos1, v_pos2) / 2.0
        h_cm = dist_form(h_pos1, h_pos2) / 2.0

        # return veritcal and horizontal scale
        return v_cm, h_cm

    # chapter 2: measuring the leaf
    # step 2: find presence of veins to assist with step 3

    def measure_veins(
            self,
            adapthist_clip_limit=ADAPTHIST_CLIP_LIMIT,
            denoise_weight=DENOISE_WEIGHT,
            canny_sigma=CANNY_SIGMA,
            hough_params=HOUGH_PARAMS,
            min_total_lines=MIN_TOTAL_LINES,
            min_center_lines=MIN_OTHER_LINES,
            flat_line_margin=FLAT_LINE_DEGREES,
            midrib_margin_percent=MIDRIB_MARGIN_PERCENT,
            min_lines_in_group=MIN_OTHER_LINES):
        """
        Measures the veins in a leaf. Split into several methods.

        The parameters for the method is disgusting, but it avoids using global variables.

        :param adapthist_clip_limit:
        :param denoise_weight:
        :param canny_sigma:
        :param hough_params:
        :param min_total_lines:
        :param min_center_lines:
        :param flat_line_margin:
        :param midrib_margin_percent:
        :param min_lines_in_group:
        :return: {
            'canny edges': vein_edges,
            'hough above': [line for line, slope, angle in above],
            'hough below': [line for line, slope, angle in below],
            'hough center': midrib_lines,
            'midvein': np.average([[p0[1], p1[1]] for p0, p1 in midrib_lines]),
            'center range': midrib_range,
            'midrib lin approx': lin_approx,
        }
        """

        # get the lines from the leaf
        lines, vein_edges = self.__measure_veins_get_lines__(
            adapthist_clip_limit=adapthist_clip_limit,
            denoise_weight=denoise_weight,
            canny_sigma=canny_sigma,
            hough_params=hough_params)

        # get the lines of the midrib
        midrib_lines, midrib_range = self.__measure_veins_get_middle__(
            lines=lines,
            min_total_lines=min_total_lines,
            min_center_lines=min_center_lines,
            flat_line_margin=flat_line_margin,
            midrib_margin_percent=midrib_margin_percent)

        # get lines above and below midrib
        above, below = self.__measure_veins_get_above_and_below__(
            lines=lines,
            midrib_lines=midrib_lines,
            min_lines_in_group=min_lines_in_group)

        # get the angle of the largest group of veins for above and below
        # the angle is put in context of the midrib's angle for continuity

        above_lines = [g[0] for g in above]
        below_lines = [g[0] for g in below]

        above_grouped = self.__measure_veins_group_veins__(above_lines)
        below_grouped = self.__measure_veins_group_veins__(below_lines)

        above_group_sums = [np.sum(group) for group in above_grouped]
        above_largest_group = above_grouped[above_group_sums.index(np.max(above_group_sums))]

        below_group_sums = [np.sum(group) for group in below_grouped]
        below_largest_group = below_grouped[below_group_sums.index(np.max(below_group_sums))]

        # get the midrib angle
        midrib_angle = 0.0
        for line in midrib_lines:
            midrib_angle += line_to_angle(line)
        midrib_angle /= len(midrib_lines)

        # get the angles of lines above and below midrib

        above_angle = 0.0
        for line in above_largest_group:
            above_angle += line_to_angle(line)
        above_angle /= len(above_largest_group)

        below_angle = 0.0
        for line in below_largest_group:
            midrib_angle += line_to_angle(line)
        midrib_angle /= len(below_largest_group)

        # put angles in context of midrib (subtract midrib angle from angle)
        above_angle -= midrib_angle
        above_angle = normalize_angle(above_angle)

        below_angle -= midrib_angle
        below_angle = normalize_angle(below_angle)

        # get the length of the veins by simply taking the sum of the canny of the edges
        vein_length = np.sum(vein_edges) / ((self.horizontal_cm_scale + self.vertical_cm_scale) / 2)

        # get the linear approximation
        lin_approx = self.__measure_veins_create_midrib_lin_approx__()

        # finally return all of the data gained from measuring the veins
        return {
            'canny edges': vein_edges,
            'hough above': [line for line, slope, angle in above],
            'hough below': [line for line, slope, angle in below],
            'hough center': midrib_lines,
            'midvein': np.average([[p0[1], p1[1]] for p0, p1 in midrib_lines]),
            'center range': midrib_range,
            'midrib lin approx': lin_approx,
            'angle above': above_angle,
            'angle below': below_angle,
            'vein length': vein_length
        }

    def __get_hough_params(self):
        """
        1/16/17

        This makes the hough params in relation to the array's size, since oyu cannot reference an obejct's self when
        defining parameters.

        This is to hopefully help the hough method pick up lines in the less detailed (smaller) images of the year 1
        collection.

        :return: tuple of (threshold, line_length, and line_gap)
        """

        threshold = 10  # not entirely sure what this does, but it is assumed to not be as dependant on image sizes
        length = self.leaf.shape[0]/32  # to get lines of length 50 in the new leaves on about 14 or so in the old
        line_gap = 7  # to meet in the middle between 10 and 5

        return (threshold, length, line_gap)

    def __measure_veins_get_lines__(self,
                                    adapthist_clip_limit=0.03,
                                    denoise_weight=0.2,
                                    canny_sigma=2.5,
                                    hough_params=(10, 30, 10)):
        """
        The first of the methods to measure the veins of a leaf. This gets the lines contained
        in the veins by first curating the leaf (with an adaptive histogram transformation and
        a denoising). Then, after the edges of the leaf are removed, the lines are found with
        the Hough probalistic line transformation.

        Mostly directly copy and pasted from old measure_veins method.

        :rtype: list
        :param adapthist_clip_limit:
        :param denoise_weight:
        :param canny_sigma:
        :param hough_params: [threshold, line_length, and line_gap]
        :return: lines
        """

        # set the hough params to be local to the size of the leaf
        hough_params = self.__get_hough_params()

        # equalize the leaf to help the vein detection
        try:
            equalized = equalize_adapthist(self.leaf, clip_limit=adapthist_clip_limit)

        except ValueError as e:
            tb = sys.exc_info()[2]
            traceback.print_tb(tb)
            print(e)

            raise MeasurementError(['Error at equalize_adapthist'])

        except ZeroDivisionError as e:
            tb = sys.exc_info()[2]
            traceback.print_tb(tb)
            print(e)

            raise MeasurementError(['Error at equalize_adapthist'])

        # try to remove any noise that could mess with the Hough function
        denoised = denoise_tv_chambolle(equalized, weight=denoise_weight, multichannel=True)

        # # make a copy of the array and set it as the bitmap
        # leaf_bitmap = denoised.copy()
        #
        # # # only show leaf that is not background with threshold
        # for row_count, row in enumerate(binary_dilation(self.leaf_bin.astype(bool))):
        #     for column_count, pixel in enumerate(row):
        #         if not pixel:
        #             leaf_bitmap[row_count, column_count] = 1

        # use a numpy masked array to get only the grayscale leaf and no background
        leaf_bitmap = np.ma.masked_array(denoised.copy(), mask=np.bool_(self.leaf_bin))

        # find the edges with the Canny method
        # sigma is set arbitrarily
        edges = canny(leaf_bitmap, sigma=canny_sigma)

        # try to make Canny's result more complete by closing gaps in the edges
        edges = binary_closing(edges)

        # remove the perimeter of the leaf, only leaving the veins inside
        vein_edges = edges - np.logical_and(edges, -binary_erosion(self.leaf_bin))

        # try and find all possible lines within the leaf
        # threshold, line_length, and line_gap all set through trial and error
        threshold, line_length, line_gap = hough_params
        lines = probabilistic_hough_line(vein_edges,
                                         threshold=threshold,
                                         line_length=line_length,
                                         line_gap=line_gap)

        # return the lines found
        return lines, vein_edges

    def __measure_veins_get_middle__(self,
                                     lines,
                                     min_total_lines=MIN_TOTAL_LINES,
                                     min_center_lines=MIN_OTHER_LINES,
                                     flat_line_margin=FLAT_LINE_DEGREES,
                                     midrib_margin_percent=0.05):
        """

        Gets the line segments of the midrib

        :param lines: all lines of the hough method
        :param min_total_lines: fail leaf if not enough total lines
        :param min_center_lines: fail leaf if not enough center lines
        :param flat_line_margin: degrees of which a line is considered flat
        :param midrib_margin_percent: percent of the leaf's width that is considered midrib
        :return: midrib lines
        """

        # fail the leaf if there are less than the minimum total number of lines
        if len(lines) < min_total_lines:
            # raise a error
            raise MeasurementError(["Too few level lines in leaf. Only {0} when {1} needed.".format(
                len(lines), min_total_lines)])

        # filter lines that are within the margin of degrees determined by +- FLAT_LINE_DEGREES
        level_lines = []

        # find level lines, which have a good chance of representing the midrib
        for l in lines:
            # append leaf if within margin
            if -flat_line_margin < line_to_angle(l) < flat_line_margin:
                level_lines.append(l)

        # get the median level line that will hopefully be the midrib.
        approx_midrib_y = np.median([np.average([p0[1], p1[1]]) for p0, p1 in level_lines])

        # calculate the range in which a line will be considered part of the midvein
        midrib_range = (
            approx_midrib_y - self.leaf_bin.shape[1] * midrib_margin_percent,  # upper bound
            approx_midrib_y + self.leaf_bin.shape[1] * midrib_margin_percent  # lower bound
        )

        # collect all lines completely within the range
        midrib_lines = []

        # for all lines that are level
        for p0, p1 in level_lines:  # this could be changed to all lines to see what would happen
            # if two points of line are both within the range, save it
            if midrib_range[0] < p0[1] < midrib_range[1] and midrib_range[0] < p1[1] < midrib_range[1]:
                midrib_lines.append((p0, p1))

        # get the approximate center of the leaf based on the lines in the midrib
        # prone to shifting where there are more level lines, not the actual midrib
        center_y = np.average([[p0[1], p1[1]] for p0, p1 in midrib_lines])
        center_x = np.average([[p0[0], p1[0]] for p0, p1 in midrib_lines])
        # center_slope = np.average([get_slope(p0, p1) for p0, p1 in center_lines])

        # fail the leaf if there are less than 8 level lines
        if len(midrib_lines) < min_center_lines:
            raise MeasurementError(
                ["Too few CENTER lines in leaf. Only {0} when {1} needed.".format(
                    len(midrib_lines), min_center_lines)])

        # calculate average slope by going to degrees, averaging, then converting back to slope
        degrees_measures = []
        for p0, p1 in midrib_lines:
            # convert line to degrees and append to list
            degrees_measures.append(line_to_angle((p0, p1)))

        # average the list
        avg_degrees = np.average(degrees_measures)

        # calculate the slope based on the degrees (more reliable than directly averaging slopes)
        avg_slope = tan(radians(avg_degrees))

        # parameters for the linear approximation if the midrib (for point slope form)
        # y - center_y = avg_slope * (x - center_x)
        self.midrib_func = [avg_slope, center_x, center_y]

        return midrib_lines, midrib_range

    def __measure_veins_get_above_and_below__(self,
                                              lines,
                                              midrib_lines,
                                              min_lines_in_group=8):
        """
        A method to get all lines above and below the midrib in two groups
        :param lines: list of all lines
        :param midrib_lines: list of lines in the midrib
        :param min_lines_in_group: minimum number of lines allowed in a group (above or below)
        :return: above, below (list of lines)
        """
        # Since the midrib as well as the center line has been found, the leaf can now be separated
        # based on the line above and below the midrib's center y value. This will deal with non-flat
        # lines that are not in the midrib.

        # get the approximate center of the leaf based on the lines in the midrib
        # prone to shifting where there are more level lines, not the actual midrib
        center_y = np.average([[p0[1], p1[1]] for p0, p1 in midrib_lines])

        # separate lines based on above and below center line
        above = []
        below = []

        # iterate through all lines
        for l in lines:
            # separate into points
            p0, p1 = l

            # check if the line is not a midrib line
            if l not in midrib_lines:
                # if both points' y values are below center y
                if center_y <= p0[1] and center_y <= p1[1]:
                    below.append([l, line_to_slope(p0, p1), line_to_angle(l)])

                # else if above center
                elif center_y >= p0[1] and center_y >= p1[1]:
                    above.append([l, line_to_slope(p0, p1), line_to_angle(l)])

        #########################################
        #                  ABOVE                #
        #########################################

        # # step 1:
        # # filter the lines that are above the center and have a negative slope (pointing towards the midrib)
        # # this assumes that all veins in the leaf point outwards (rather than in, towards the midrib)
        # for l, slope, angle in above:
        #     # remove lines with a negative slope
        #     if not slope > 0.:
        #         above.remove([l, slope, angle])
        #
        # # step 2:
        # # now filter based on lines that are close to the median line's degrees measure
        # # calculate median degrees measure
        # median_degrees = np.median([[angle] for l, slope, angle in above])
        #
        # # calculate margin
        # margin = [median_degrees - degrees_margin,
        #           median_degrees + degrees_margin]
        #
        # # filter by margin
        # for l, slope, angle in above:
        #     # remove if the line doesn't satisfy
        #     if not margin[0] < angle < margin[1]:
        #         above.remove([l, slope, angle])

        # fail the leaf if there are less than 8 level lines
        if len(above) < min_lines_in_group:
            raise MeasurementError(
                ["Too few ABOVE lines in leaf. Only {0} when {1} needed.".format(
                    len(above), min_lines_in_group)])

        #########################################
        #                  BELOW                #
        #########################################

        # # step 1:
        # # filter the lines that are below the center and have a postive slope (pointing towards the midrib)
        # # this assumes that all veins in the leaf point outwards (rather than in, towards the midrib)
        # for l, slope, angle in below:
        #     # remove lines with a positive slope
        #     if not slope < 0.:
        #         below.remove([l, slope, angle])
        #
        # # step 2:
        # # now filter based on lines that are close to the median line's degrees measure
        # # calculate median degrees measure
        # median_degrees = np.median([[angle] for l, slope, angle in below])
        #
        # # calculate margin
        # margin = [median_degrees - degrees_margin,
        #           median_degrees + degrees_margin]
        #
        # # filter by margin
        # for l, slope, angle in below:
        #     # remove if the line doesn't satisfy
        #     if not margin[0] < angle < margin[1]:
        #         below.remove([l, slope, angle])

        # fail the leaf if there are less than 8 level lines
        if len(below) < min_lines_in_group:
            raise MeasurementError(
                ["Too few BELOW lines in leaf. Only {0} when {1} needed.".format(
                    len(below), min_lines_in_group)])

        return above, below

    def __measure_veins_create_midrib_lin_approx__(self):
        """
        create method to plot midrib approximation
        :return: function of x
        """
        m, x0, y0 = self.midrib_func

        def midrib_lin_approx(x):
            """
            This method plots the midrib line. Designed to be returned from the measure_veins function
            as a way to visualize the midrib.
            :param x: the x-value for a line
            :return: y-value correlating to the x-value
            """
            y = int(m * (x - x0) + y0)
            return y

        return midrib_lin_approx

    @staticmethod
    def __measure_veins_group_veins__(lines):
        """
        A method to group lines of similar angles.
        :param lines: above or below lists
        :return:
        """

        # get angles of lines
        # sort so indexes correspond
        lines_with_angles = {line_to_angle(line): line for line in lines}

        # cluster angles
        clustered_angles = cluster_by_diff([k for k in lines_with_angles.keys()], 15.0)

        # translate the new angles back to the lines in thier clusters
        clustered_lines = []
        for cluster in clustered_angles:
            new_cluster = []
            for angle in cluster:
                new_cluster.append(lines_with_angles[angle])
            clustered_lines.append(new_cluster)

        return clustered_lines

    #
    #
    #
    #
    #
    #
    #
    #
    # step 3: measure length based on midrib
    def measure_length(self):
        """
        Measures the length of the leaf using the center line provided by
        measure_veins() and the scale provided by get_scale().

        Assumes measure_veins() has already been ran.
        :return: length in cm
        """
        # get the midrib linear approximation from the vein_measure
        midrib_lin_approx = self.vein_measure['midrib lin approx']

        # initialize points
        p0 = (None, None)
        p1 = (None, None)

        # get the endpoints of the leaf based on the linear approximation
        for x in range(0, self.leaf_bin.shape[1]):
            # calculate the y-value corresponding to a point in the leaf
            y = midrib_lin_approx(x)

            # get the boolean image's value at the current point (in the binary leaf)
            point_value = self.leaf_bin[y][x]

            # if the value is at the leaf, save the point and break the loop
            if point_value:
                p0 = (x, y)
                break

        # repeat the same process from the other side of the image
        for x in range(0, self.leaf_bin.shape[1])[::-1]:
            # calculate the y-value corresponding to a point in the leaf
            y = midrib_lin_approx(x)

            # get the boolean image's value at the current point (in the binary leaf)
            point_value = self.leaf_bin[y][x]

            # if the value is at the leaf, save the point and break the loop
            if point_value:
                p1 = (x, y)
                break

        # calculate the distance between the points
        length = dist_form(p0, p1, v_scale=self.vertical_cm_scale, h_scale=self.horizontal_cm_scale)

        # save the endpoints of the leaf
        self.endpoints = (p0, p1)

        # return the length that was calculated
        return length

    # step 4: measure perimeter of leaf

    def measure_perimeter(self):
        """
        Measures the perimeter of the leaf based on the sum of the pixels
        in the Canny method divided by the scale.

        :return: perimeter (in cm.)
        """

        # use the numpy method for calculating perimeter to find perimeter
        p = perimeter(self.leaf_bin)

        # return p in raw form and in respect to the horizontal and vertical scales
        return p, p / self.horizontal_cm_scale, p / self.vertical_cm_scale

    # step 5: measure by splitting lines

    def measure_contours(self, levels=3):
        """
        This measures the width of the leaf based on where the widest portion of the leaf is. Once the widest part is
        found, the two sides created by dividing the leaf by this measurement are then measured somewhat recursively in
        fractions (1/2s, 1/4s, 1/8s, maybe etc.)

        Once divided, the widths of the sections are measured (at the divisions). After that, the slopes are measured
        between points on the sections.

        :param levels: int, how many 'levels' should the leaf be split into (2^levels is how many resulting sections of
        leaves will be left)
        :return:
        """

        # STEP 1: Find the widest width on the leaf

        # sum all columns in leaf
        column_sums = [int(np.sum(column)) for column in self.leaf_bin.T]

        # find max and max's index
        max_width = np.max(column_sums)
        max_width_index = column_sums.index(max_width)

        # find left and right bounds of leaf based on presence (not endpoints)
        left_bound = None
        for count, i in enumerate(column_sums):
            if i is not 0:
                left_bound = count

        right_bound = None
        for count, i in [[count, j] for count, j in enumerate(column_sums)][::-1]:
            if i is not 0:
                right_bound = count

        # STEP 2: Split leaf into halves

        # start splitting to levels
        half_1 = self.leaf_bin[:, left_bound:max_width_index].copy()
        half_2 = self.leaf_bin[:, max_width_index:right_bound].copy()

        # each level of levels dict:
        # order, slice of binary leaf, bounds

        levels_dict = {
            1:
                [
                    [0, half_1, (left_bound, max_width_index)],
                    [1, half_2, (max_width_index, right_bound)]
                ]
        }

        # STEP 3: Split the leaf into more sections

        # iterate through levels, measuring appropriate amounts
        for i in range(1, levels):
            # according to how the range() works, i will be the key for the sections that need to be processed by
            # __find_half_measurement__(), and i + 1 will be the next level. It should build itself.

            # this list serves to temporarily store all values of the new splits before sorting
            unsorted_split = []

            # split each fraction of a leaf in a level into two more fractions
            for order, array, bounds in levels_dict[i]:
                # split array
                array1, array2, bounds1, bounds2 = self.split_array_by_bounds(array, bounds)

                # append the new split into a temporary list
                unsorted_split.append([
                    order,
                    [0, array1, bounds1],
                    [1, array2, bounds2],
                ])

            # now sort the list
            sorted_split = []
            for count, fraction in enumerate(unsorted_split):
                # unpack data at index
                parent_order, array_data1, array_data2 = fraction

                # prepare order values
                child_order = parent_order * 2

                # repack arrays into the format that will actually go into the dictionary
                array1_data_ordered = [
                    child_order + array_data1[0],  # order
                    array_data1[1],  # actual array
                    array_data1[2],  # bounds
                ]
                array2_data_ordered = [
                    child_order + array_data2[0],  # order
                    array_data2[1],  # actual array
                    array_data2[2],  # bounds
                ]

                # append to sorted_split
                sorted_split.append(array1_data_ordered)
                sorted_split.append(array2_data_ordered)

            # add to dict
            levels_dict[i + 1] = sorted_split

        # STEP 4: Measure the widths at the middle lines.

        # first, get all the bounds
        bounds_list = [l[2] for l in levels_dict[levels]]

        # turn the range-style bounds into numbers
        x_splits = sorted([int(bounds_list[i][0]) for i in range(1, len(bounds_list))])

        # now find the y-values at the given x-values
        slice_dict = {}
        widths_dict = {}
        for count, x in enumerate(x_splits):
            # get a column of values
            column = self.leaf_bin.T[x].tolist()

            # find the indexes of the first signs of pixels
            y0 = column.index(1)
            y1 = len(column) - column[::-1].index(1)

            # save it to a dict
            slice_dict[count] = ((x, y0), (x, y1))
            widths_dict[count] = (y1 - y0) / self.vertical_cm_scale

        # now find slopes between top points and slopes between bottom points

        # separate slice_dict by top and bottom points
        top_points = {k: slice_dict[k][0] for k in slice_dict.keys()}
        bottom_points = {k: slice_dict[k][1] for k in slice_dict.keys()}

        # measure angles between lines
        angles_dict = {}
        for i in range(1, len(slice_dict)):
            # top
            p0 = top_points[i - 1]
            p1 = top_points[i]
            alpha = normalize_angle(line_to_angle((p0, p1)))

            # bottom
            p0 = bottom_points[i - 1]
            p1 = bottom_points[i]
            beta = normalize_angle(line_to_angle((p0, p1)))

            # average the angles in a smart way
            beta -= 360
            beta = abs(beta)
            gamma = (alpha + beta) / 2

            # save the angle to angles dict
            angles_dict[i] = gamma

        # return the dictionary of contours and angles
        return slice_dict, widths_dict, angles_dict

    @staticmethod
    def split_array_by_bounds(array, x_bounds):
        """
        This function works closely with the measure_contours() function. This is meant to stop the copy and pasting of
        code and make it easier to take more or less measurements if needed.

        This works between two x boundaries to find the midpoint between to two (basically the average). Then, the array
        is split into two more arrays around the new x boundaries.

        :param array: nparray, a section of a leaf image
        :param x_bounds: tuple containing the x-values of the beginning and end of the leaf
        :return: array1, array2, x_bounds1, x_bounds2 (similar to inputs)
        """

        # find midpoint to split leaf by
        midpoint_x = np.average(x_bounds)

        # create bounds
        x_bounds1 = (int(x_bounds[0]), int(midpoint_x))
        x_bounds2 = (int(midpoint_x), int(x_bounds[1]))

        # split arrays
        array1 = array[:, x_bounds1[0]:x_bounds1[1]].copy()
        array2 = array[:, x_bounds2[0]:x_bounds2[1]].copy()

        # return values
        return array1, array2, x_bounds1, x_bounds2

    def measure_surface_variability(self):
        """
        Measures the mean difference between columns in a leaf in order to find a value of how much the leaf changes
        widths on average.
        
        Also returns a median for comparison.
        :rtype: list
        :return: float mean and median (float because it will be in cms)
        """

        # sum the columns
        column_sums = [int(np.sum(column)) for column in self.leaf_bin.T if np.any(column)]

        # remove zeros
        for count, n in enumerate(column_sums):
            if n == 0:
                column_sums.remove(count)

        # find differences
        column_diffs = []

        for count, n in enumerate(column_sums[1:]):
            diff = abs(column_sums[count] - column_sums[count - 1])
            column_diffs.append(diff)

        # calculate pixel values
        mean_p = np.average(column_diffs)
        median_p = np.median(column_diffs)

        # now in terms of cms
        mean_cm_h = mean_p / self.horizontal_cm_scale
        mean_cm_v = mean_p / self.vertical_cm_scale

        median_cm_h = median_p / self.horizontal_cm_scale
        median_cm_v = median_p / self.vertical_cm_scale

        # return values
        return (mean_cm_h + mean_cm_v) / 2, (median_cm_h + median_cm_v) / 2


class MeasurementError(Exception):
    def __init__(self, args):
        """
        A custom exception for raising custom errors
        :param args
        """
        self.args = args

class AdaptHistError(Exception):
    def __init__(self, args):
        """
        A custom exception for raising custom errors
        :param args
        """
        self.args = args


def line_to_unit_vector(line):
    """
    Converts a line to a unit vector.
    :param line: ((x0, y0), (x1, y1))
    :return: ((a, b), (x0, y0))
    """

    # unpack line
    p0, p1 = line
    x0, y0 = p0
    x1, y1 = p1

    # get vector components
    a = x1 - x0
    b = y1 - y0

    # get vector magnitude
    mag = (a ^ 2.0 + b ^ 2.0) ^ 0.5

    # make vector tuple as listed at beginning
    vec = ((a / mag, b / mag), p0)

    # return vector
    return vec


def cluster_by_diff(data, max_gap):
    """
    a function that clusters numbers based on their differences

    based off of a stacktrace answer:
    http://stackoverflow.com/a/14783998
    :param data: any list of floats or ints
    :param max_gap: the largest gap between numbers until starting a new cluster
    :return: nested list
    """
    # since the data needs to be sorted to determine useful differences, sort the data
    data.sort()

    # initialize the nest list
    groups = [[data[0]]]

    # iterate through data
    for x in data[1:]:
        # compare the difference of the first value of the data and the last entry in the groups to the max gap
        if abs(x - groups[-1][-1]) <= max_gap:
            # not larger than gap, append to last group
            groups[-1].append(x)
        else:
            # make new group if larger
            groups.append([x])

    return groups


def rotate_line_about_point(line, point, degrees):
    """
    added 161205

    This takes a line and rotates it about a point a certain number of degrees.

    For use with clustering veins.

    :param line: tuple contain two pairs of x,y values
    :param point: tuple of x, y
    :param degrees: number of degrees to rotate by
    :return: line (now rotated)
    """

    # point will serve as axis
    axis = point

    # unpack line
    p0, p1 = line

    # and get the line's degrees and length
    line_deg = line_to_angle(line)
    d = (abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
    line_length = sqrt(d[0] ^ 2 + d[1] ^ 2)

    # calculate radius between points and axis
    d = (abs(p0[0] - axis[0]), abs(p0[1] - axis[1]))
    r0 = sqrt(d[0] ^ 2 + d[1] ^ 2)
    # r1 = float((p1[0] - axis[0]) ^ 2 + (p1[1] - axis[1]) ^ 2) ^ 0.5

    # find degrees that first line is above x-axis
    p0_deg = line_to_angle((axis, p0))

    # now rotate line one to be level to degrees
    p0_cos = cos(degrees * (pi / 180.0))
    p0_sin = sin(degrees * (pi / 180.0))

    p0_n = (r0 * p0_cos, r0 * p0_sin)

    # and move p1 to be in respect to p0
    new_deg = line_deg - p0_deg

    # normalize degrees
    while new_deg > 360:
        new_deg -= 360
    while new_deg < 0:
        new_deg += 360

    # get second point of line now since all variables are known
    p1_cos = cos(new_deg * (pi / 180.0))
    p1_sin = sin(new_deg * (pi / 180.0))

    # get new p1
    p1_n = (p1_cos * line_length + p0_n[0], p1_sin * line_length + p0_n[1])

    # return new line
    return p0_n, p1_n


def normalize_angle(deg):
    """
    Take an angle in degrees and return it as a value between 0 and 360
    :param deg: float or int
    :return: float or int, value between 0 and 360
    """
    angle = deg
    while angle > 360:
        angle -= 360

    while angle < 360:
        angle += 360

    return angle
