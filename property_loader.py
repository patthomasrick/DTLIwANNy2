import os

# ~/.pyenv/versions/3.5.2/bin/python

"""
property_loader.py: loads the right property file from the image folder
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "8/14/16"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"

PROP_DIR = os.path.join(os.getcwd(), 'input-images', 'properties')


def load_properties(bin_nom, p=PROP_DIR):
    """
    from bin_nom, the function will find the corresponding bin_nom.txt file in properties, load
    the contents of the file, parse the contents, and return them in a nice format.
    :param bin_nom: str in format "Genus species"
    :return: list of props.
    """
    fname = os.path.join(p, '{0}.txt'.format(bin_nom))
    f = open(fname, 'r')
    r_list = []

    for line in f:
        s = line.strip()
        if ':' in s:
            wide, narrow = s.split(':')
            for n in narrow.split(','):
                r_list.append('{0}-{1}'.format(wide, n))
        else:
            r_list.append(s)

    return r_list