from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.io import imshow, show
from skimage.morphology import binary_erosion, binary_closing
from skimage.restoration import denoise_tv_chambolle

from ruler import Ruler

"""
main-demo.py

This provides a demo presentation for science fair.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "01/28/17"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"


# noinspection PyAttributeOutsideInit
class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

        self.r = Ruler()

    def create_widgets(self):
        """
        Creates all the widgets for tk
        :return: None
        """
        # buttons
        self.path_box = Text(self, height=1, width=80)
        self.path_box.insert(
            END,
            'input-images/Leaf Collection 160711/Acer saccharum/IMG_6749.JPG'
        )
        self.path_box.pack(side='top')

        self.show_leaf = Button(
            self,
            text='Show image',
            command=self.show_leaf_cmd
        )
        self.show_leaf.pack(side='top')

        self.show_split = Button(
            self,
            text='Split and crop leaf and scale',
            command=self.show_split_cmd
        )
        self.show_split.pack(side='top')

        self.show_threshold = Button(
            self,
            text='Threshold image',
            command=self.show_threshold_cmd
        )
        self.show_threshold.pack(side='top')

        self.show_veins = Button(
            self,
            text='Show veins and vein lines',
            command=self.show_veins_cmd
        )
        self.show_veins.pack(side='top')

    def show_leaf_cmd(self):
        """
        Loads and shows a leaf image.
        :return: none
        """
        # load current leaf path
        self.r.load_new_image(self.path_box.get('1.0', 'end-1c').strip(), no_scale=False)

        imshow(self.r.img)
        show()

    def show_split_cmd(self):
        """
        Splits and crops and shows a leaf image.
        :return: none
        """
        # load current leaf path
        self.r.load_new_image(self.path_box.get('1.0', 'end-1c').strip(), no_scale=False)

        # displaying data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes.ravel()
        ax[0].imshow(self.r.scale, cmap=plt.cm.gray)
        ax[0].set_title('Scale')
        ax[1].imshow(self.r.leaf, cmap=plt.cm.gray)
        ax[1].set_title('Leaf')

        plt.show()

    def show_threshold_cmd(self):
        """
        Splits and crops and shows a leaf image.
        :return: none
        """
        # load current leaf path
        self.r.load_new_image(self.path_box.get('1.0', 'end-1c').strip(), no_scale=False)

        # displaying data
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes.ravel()
        ax[0].imshow(-self.r.scale_bin, cmap=plt.cm.gray)
        ax[0].set_title('Scale')
        ax[1].imshow(-self.r.leaf_bin, cmap=plt.cm.gray)
        ax[1].set_title('Leaf')

        plt.show()

    def show_veins_cmd(self):
        """
        Splits and crops and shows a leaf image.
        :return: none
        """
        # load current leaf path
        self.r.load_new_image(self.path_box.get('1.0', 'end-1c').strip(), no_scale=False)

        # displaying data
        img = self.r.leaf
        hough_center = self.r.vein_measure['hough center']
        hough_above = self.r.vein_measure['hough above']
        hough_below = self.r.vein_measure['hough below']
        hough_range = self.r.vein_measure['center range']
        midrib_line = self.r.vein_measure['midrib lin approx']
        length = self.r.length

        equalized = equalize_adapthist(self.r.leaf, clip_limit=0.03)
        denoised = denoise_tv_chambolle(equalized, weight=0.2, multichannel=True)
        leaf_bitmap = np.ma.masked_array(denoised.copy(), mask=np.bool_(self.r.leaf_bin))
        edges = canny(leaf_bitmap, sigma=2.5)
        edges = binary_closing(edges)
        vein_edges = edges - np.logical_and(edges, -binary_erosion(self.r.leaf_bin))


        print(hough_above)

        # displaying data with pyplot and matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        ax = axes.ravel()
        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_title('Leaf')
        row, col = img.shape
        ax[1].imshow(vein_edges, cmap=plt.cm.gray)
        ax[1].set_title('Canny method')
        ax[2].axis((0, col, row, 0))
        ax[2].imshow(-img, cmap=plt.cm.gray)
        for line in hough_center:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), 'b')
        for line in hough_above:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), 'g')
        for line in hough_below:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), 'r')
        ax[2].plot((0, img.shape[1]), (hough_range[0], hough_range[0]), 'b--')
        ax[2].plot((0, img.shape[1]), (hough_range[1], hough_range[1]), 'g--')
        ax[2].plot((0, img.shape[1]), (midrib_line(0), midrib_line(img.shape[1])))
        ax[2].set_title('Hough lines over leaf')

        plt.show()


if __name__ == '__main__':
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry('{0}x{1}'.format(500, 150))
    app = Application(master=root)
    app.mainloop()
