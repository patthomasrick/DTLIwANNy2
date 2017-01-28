
from os.path import basename

"""
1/11/17

tracer.py: Provides a class to have toggleable log messages for easier tracing of errors.
"""

__author__ = "Patrick Thomas"
__credits__ = ["Patrick Thomas", "Rick Fisher"]
__version__ = "1.0.0"
__date__ = "01/11/17"
__maintainer__ = "Patrick Thomas"
__email__ = "pthomas@mail.swvgs.us"
__status__ = "Development"


class Tracer:
    def __init__(self, mode=0):
        # modes can be:
        # 0 off
        # 1 minimal
        # 5 verbose
        self.mode = mode

    def print(self, message, mode, file):
        """
        1/11/17

        Prints message based on mode
        :param message: str
        :param mode: num, custom range
        :return: message
        """
        if mode <= self.mode:
            print(basename(file) + ':', message)

        return message