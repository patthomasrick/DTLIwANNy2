from os import listdir
from os.path import isfile, join

# ~/.pyenv/versions/3.5.1/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class

"""
species_sort.py: A class to help sort leaves by their species.

Provides a classs to define a species, which has the capabilities of
returning basic information about the species as well as where the species
is stored on the hard drive.
"""

__author__ = "Patrick Thomas"
__version__ = "1.0"
__date__ = "6/28/16"


class Species:
    """
    The class to define species. Allows the storage and retrieval of species names such as binomial nomenclature.
    """

    def __init__(self, genus, species, properties=None):
        """
        Initializes the species class by getting the two components of the binomial nomenclature name.
        :rtype: Species()
        :param genus: str of genus, e.g. "Quercus"
        :param species: str of species, e.g. "alba"
        """
        if properties is None:
            properties = []
        self.genus = genus
        self.species = species
        self.bin_nom = '{0} {1}'.format(genus.title(), species)
        self.load_locations = {}
        self.properties = properties

    def __repr__(self):
        """
        Returns the species itself
        :return: self
        """
        return self.bin_nom

    def __str__(self):
        """
        Returns the str binomial nomenclature
        :return: self.bin_nom
        """
        return self.bin_nom

    def add_load_location(self, location_path, location_name):
        """
        Adds a location to load from to the self.load_locations var
        :param location_path: path of the image folder for the species
        :param location_name: for easy retrieval
        :return:
        """
        self.load_locations[location_name] = location_path

    def add_properties(self, *args):
        """
        Adds all of the args as properties of the species, such as lobulate or asymmetrical
        :param args:
        :return:
        """
        for s in args:
            if s not in self.properties:
                self.properties.append(s)

    def get_leaf_paths(self):
        if self.load_locations == {}:
            print('Cannot load images from "{0}", no paths are given'.format(self.bin_nom))
            return None
        else:
            all_loc_files = []
            for loc_key in self.load_locations.keys():
                loc_path = self.load_locations[loc_key]
                all_loc_files.extend(
                    [join(loc_path, f) for f in listdir(loc_path) if isfile(
                        join(loc_path, f))])
            return all_loc_files
