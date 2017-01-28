def measure_contour_widths(self, levels=3):
    """
    This measures the width of the leaf based on where the widest portion of the leaf is. Once the widest part is
    found, the two sides created by dividing the leaf by this measurement are then measured somewhat recursively in
    fractions (1/2s, 1/4s, 1/8s, maybe etc.)

    :param levels: int, how many 'levels' should the leaf be split into (2^levels is how many resulting sections of
    leaves will be left)
    :return:
    """

    #########################################
    #               MOST WIDE               #
    #########################################

    # as of now, this part of the method just finds the tallest column and says that is the widests portion of the
    # leaf, ignoring any rotation of the leaf that may be present.

    # this could possibly be fixed by measuring for the tall column in each half of the leaf (above and below the
    # midrib), but this does not work well with the propotions measuring of the sides.

    # also the other method would be more prone to mistakes that would be hard to pin down, where by just measuring
    # the tallest column once would be rather more robust (but less precise)

    # list to contain all slices of the leaf that are found
    all_widths = []

    # bounds contain all parts of the image that has some pixels
    bounds = {}

    # iterate through image (transformed 90 degrees for ease)
    for x, leaf_slice in enumerate(self.leaf_bin.T):
        if np.sum(leaf_slice) != 0:  # if the slice isn't empty...
            # initialized to None since there are some pixels in this region (as found
            # by previous if-statement)
            top_boundary = None
            bottom_boundary = None

            # iterate from the top and find the first pixel
            for pos_y, pixel in enumerate(leaf_slice):
                # if there is a pixel, it is now the upper bound
                if pixel:
                    top_boundary = pos_y

            # iterate from the bottom and find the last (first from the bottom) pixel
            for pos_y, pixel in enumerate(leaf_slice[::-1]):
                # if there is a pixel, it is now the lower bound
                if pixel:
                    bottom_boundary = len(leaf_slice) - pos_y

            # save the width
            all_widths.append(bottom_boundary - top_boundary)
            bounds[x] = (bottom_boundary, top_boundary)

        else:  # there are no pixels in the slice...
            all_widths.append(0)

    # find the greatest difference in bounds, aka greatest width
    diffs = [[l - h] for l, h in bounds.values()]
    max_width = np.max(diffs)

    sums = [np.sum(row) for row in self.leaf_bin.T]
    max_width = np.max(sums)
    x_widest = sums.index(max_width)

    # # calculate the median x of all slices that match the greatest width
    # widest_bounds = []
    # for x in bounds.keys():
    #     # make code more readable
    #     lower, upper = bounds[x]
    #
    #     # calculate difference
    #     diff = lower - upper
    #
    #     # if the difference matches the maximum, then append to widest bounds dict
    #     if diff == max_width:
    #         widest_bounds.append(x)
    #         # actual bounds are irrelevant since we know the width and we will have the x-value
    #
    # # find the median of the x-values
    # x_widest = np.median(widest_bounds)

    # get the endpoints' x-values for readability
    endpoint_left_x = self.endpoints[0][0]
    endpoint_right_x = self.endpoints[1][0]

    # position of the greatest width will be given to ANN as a proportion to the leaf's length
    x_widest_proportion = float(x_widest - endpoint_right_x) / self.length

    # Now the leaf is divided into the first 2 sections for each half
    half_1 = self.leaf_bin[:, 0:x_widest]
    half_2 = self.leaf_bin[:, x_widest:]

    # split halves into (levels-1) chunks
    # have a dictionary listing the chunks by level
    # e.g.:
    # levels_dict = {
    #     level: [order, chunks, bounds],
    #     ...
    # }
    levels_dict = {
        1: [[0, half_1, [endpoint_left_x, x_widest_proportion]],
            [1, half_2, [endpoint_right_x, x_widest_proportion]]]
    }

    for i in range(1, levels):
        # according to how the range() works, i will be the key for the sections that need to be processed by
        # __find_half_measurement__(), and i + 1 will be the next level. It should build itself.
        unsorted_split = []
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
        for j in range(0, len(unsorted_split)):
            # unpack data at index
            parent_order, array_data1, array_data2 = unsorted_split[j]

            # prepare order values
            child_order = parent_order * 2

            # repack arrays
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

    return levels_dict


@staticmethod
def __find_half_measurement__(array, x_bounds):
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
    x_bounds1 = (x_bounds[0], midpoint_x)
    x_bounds2 = (midpoint_x, x_bounds[1])

    # split arrays
    array1 = array[:, x_bounds1[0]:x_bounds1[1]]
    array2 = array[:, x_bounds2[0]:x_bounds2[1]]

    # return values
    return array1, array2, x_bounds1, x_bounds2