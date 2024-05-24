import numpy as np

def repeat_and_fill(array, subset):

    array = np.reshape(array, (1, len(array)))
    array = np.tile(array, (len(subset), 1))

    j = 0
    for col in range(len(array[0])):
        if np.isnan(array[0][col]):
            array[:, col] = subset[:, j]
            j += 1

    return

if __name__ == "__main__":

    array = np.array([1,np.nan,np.nan, 3, 4])
    subset = np.array([

    [0, 1],
    [1, 0]

    ])

    res = repeat_and_fill(array, subset)
