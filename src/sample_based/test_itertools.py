import itertools

import numpy as np


def yield_subset(bins, size):

    iterator = itertools.product(*bins)
    objs = np.zeros((size, len(bins)))

    l = 0
    for p in iterator:
        objs[l] = p
        l += 1
        if l >= size:
            del iterator
            return objs

    del iterator
    return objs


if __name__ == "__main__":

    bins = [[0.3839745962155614, 0.8169872981077807, 1.25],
    [1.0916876048223, 1.92084380241115, 2.75],
    [1.791960108450192, 3.270980054225096, 4.75]]

    # iterator = itertools.product(*bins)
    #
    limit = 10
    l = 0

    print((yield_subset(bins, limit)))

    print()

    singl = (itertools.product([[0.8169873, 1.25, 1.6830127]]))

    for i in singl:
        print(i)
