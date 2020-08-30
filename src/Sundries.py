import numpy as np


# percent, float, the percentage of np_array to get
# np_array, numpy array,

def get_x_percent_of_ndarray(percent, np_array):
    np_array_len = int(len(np_array) * percent)
    rating_np_indices = np.array(range(len(np_array)))
    np.random.shuffle(rating_np_indices)
    return np_array[rating_np_indices[:np_array_len], :]
