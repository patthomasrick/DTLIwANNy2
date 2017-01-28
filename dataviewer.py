from tkinter import *
from numpy import mean, std
import xml.etree.ElementTree as ETree

# ~/.pyenv/versions/3.5.2/bin/python
"""
dataviewer.py

This file is to provide a way to convieniently see all of the data
within the .xml data file.

As of now, exploring the data file is clunky as it has to be done
in a web browser and shows the information within the .xml file in
an arbitrary order.

Originial built from a modified example for tk from the Python 3.5.2
documentation on tk (https://docs.python.org/3/library/tkinter.html).
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "8/9/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"


# noinspection PyAttributeOutsideInit
class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        """
        Creates all the widgets for tk
        :return: None
        """
        # exit button
        self.quit = Button(self,
                           text="QUIT",
                           fg="red",
                           command=root.destroy)
        self.quit.pack(side="bottom")

        self.path_box = Text(self, height=1, width=80)
        self.path_button = Button(self,
                                  text="Load file",
                                  command=self.read_xml)
        self.path_box.pack(side='top')
        self.path_button.pack(side='top')
        self.path_box.insert(END, 'save-data/leaf-data.xml')

        # text display area and scrollbar block
        self.textbox = Text(self, height=50, width=140)
        self.scrollbar = Scrollbar(self, )
        self.textbox.pack(side='left')
        self.scrollbar.pack(side='right', fill=Y)
        self.textbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.textbox.yview)

    def read_xml(self):
        """
        Reads the xml file from the path at self.path_box on self.path_button
        click.
        Returns all data to self.textbox
        :return: None
        """
        tree = ETree.parse(self.path_box.get('1.0', 'end-1c').strip())
        xml_root = tree.getroot()

        buffer = []
        total_leaves = 0
        for species in xml_root:
            # get the bin nom name
            g = species.find('g').text
            s = species.find('s').text

            # sum the leaves for totals
            num_of_imgs = len(species.findall('leaf'))
            total_leaves += num_of_imgs

            # # find and compute lengths and length stats
            # lengths = []
            # for leaf in species.findall('leaf'):
            #     lengths.append(float(leaf.find('length').text))
            # avg_length = mean(lengths)
            # std_length = std(lengths)
            # len_s = str(round(avg_length, 2)).ljust(5, '0')
            # std_s = str(round(std_length, 2)).ljust(5, '0')
            #
            # # find and compute perimeter and perimeter stats
            # perimeters = []
            # perimeters_h = []
            # perimeters_v = []
            # for leaf in species.findall('leaf'):
            #     perimeters.append(float(leaf.find('p').text))
            #     perimeters_h.append(float(leaf.find('p_h').text))
            #     perimeters_v.append(float(leaf.find('p_v').text))
            # avg_perimeter = int(mean(perimeters))
            # avg_perimeter_h = int(mean(perimeters_h))
            # avg_perimeter_v = int(mean(perimeters_v))
            # s_p = str(avg_perimeter)
            # s_p_h = str(avg_perimeter_h)
            # s_p_v = str(avg_perimeter_v)
            #
            # # find and compute area and area stats
            # areas = []
            # for leaf in species.findall('leaf'):
            #     areas.append(float(leaf.find('area').text))
            # avg_perimeter = int(mean(perimeters))
            # avg_perimeter_h = int(mean(perimeters_h))
            # avg_perimeter_v = int(mean(perimeters_v))
            # s_p = str(avg_perimeter)
            # s_p_h = str(avg_perimeter_h)
            # s_p_v = str(avg_perimeter_v)

            # STATS COMPUTATIONS
            lengths = []
            perimeters = []
            areas = []

            # iterate through all leaves
            for leaf in species.findall('leaf'):
                # length
                lengths.append(float(leaf.find('length').text))

                # perimeter
                perimeters.append(float(leaf.find('p').text))

                # area
                areas.append(float(leaf.find('area').text))

            # means and stdevs
            avg_l = mean(lengths)
            avg_p = mean(perimeters)
            avg_a = mean(areas)

            std_l = std(lengths)
            std_p = std(perimeters)
            std_a = std(areas)

            # convert to strings
            str_avg_l = str(round(avg_l, 2)).ljust(5, '0')
            str_avg_p = str(round(avg_p, 2)).ljust(5, '0')
            str_avg_a = str(round(avg_a, 2)).ljust(5, '0')

            str_std_l = str(round(std_l, 2)).ljust(5, '0')
            str_std_p = str(round(std_p, 2)).ljust(5, '0')
            str_std_a = str(round(std_a, 2)).ljust(5, '0')

            buffer.append(['{0} {1}'.format(g, s),
                           num_of_imgs,
                           str_avg_l,
                           str_std_l,
                           str_avg_p,
                           str_std_p,
                           str_avg_a,
                           str_std_a,
                           ])

        lines = ["Species\t\t\t\tNum. of leaves\t\tAvg. length\t\tStdev. length\t\t" +
                 "Avg. perim.\t\tStdev. perim.\t\tAvg. area\t\tStdev. area\n"]
        for bin_nom, num, avg_l, std_l, avg_p, std_p, avg_a, std_a in sorted(buffer):
            lines.append('{0}\t\t\t\t{1}\t\t{2}\t\t{3}\t\t{4}\t\t{5}\t\t{6}\t\t{7}\n'.format(
                bin_nom,
                num,
                avg_l,
                std_l,
                avg_p,
                std_p,
                avg_a,
                std_a)
            )
        lines.append('-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n')
        lines.append('Total\t\t\t\t{0}\n\n\n'.format(total_leaves))
        text = ''
        for line in lines:
            text += line
        self.textbox.insert(END, text)

if __name__ == '__main__':
    root = Tk()
    app = Application(master=root)
    app.mainloop()
