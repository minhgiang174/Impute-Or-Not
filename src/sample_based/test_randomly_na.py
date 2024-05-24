import numpy as np

def randomly_na(data, percent_missing):

    if percent_missing > 1:
        percent_missing = percent_missing / 100

    mask = np.random.rand(data.shape[0], data.shape[1])
    mask = mask < percent_missing
    data = (np.where(mask, np.nan, data))


    return data

if __name__ == "__main__":

    data = np.array([

    [1,2,3] for i in range(20)


    ])
    print(data)

    print(data.shape)

    print(

    )

    print(randomly_na(data, 0.5))

    print()

    print(data)
