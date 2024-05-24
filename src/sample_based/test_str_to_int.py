import numpy as np



def str_to_int_labels(labels):

    strs = np.unique(labels)

    for i in range(len(strs)):
        labels = np.where(labels == strs[i], i, labels)

    return labels.astype(np.int)

if __name__ == "__main__":

    labels = np.array(["N", "O", "N", "N"])

    labels = str_to_int_labels(labels)
    print(labels)
