import numpy as np

def load_and_assert(file_path):
    # 1. Load a npy file into ndarray, assert that the loaded array has shape (*,*, 7)
    data = np.load(file_path)
    assert (len(data.shape) == 3) & (data.shape[2] == 7)
    return data


def extract_windows(data, sliding_window_size, sample_size):
    # 2. Crop the loaded array along the dimension number 1 (count starts at 0) with a sliding window of
    # length K, and take every 10th sample in dimension number 1, to an array shape of (*,*, K,
    # 7).
    sub_windows = (
        np.expand_dims(np.arange(sliding_window_size), 0) +
        np.expand_dims(np.arange(data.shape[1]
                                 - sliding_window_size + 2, step=sample_size), 0).T
    )
    return data[:, sub_windows, :]


def extract_feature(data, feature_dim):
    # Take the 2nd feature (dimension -1) to an array of shape (*,*,K),
    return data[:, :, :, feature_dim]

def crop_and_reshape(data, sliding_window_size, sample_size, feature_dim):
    # 2. Crop the loaded array along the dimension number 1 (count starts at 0) with a sliding window of
    # length K, and take every 10th sample in dimension number 1, to an array shape of (*,*, K,
    # 7).
    data = extract_windows(data, sliding_window_size, sample_size)
    # Take the 2nd feature (dimension -1) to an array of shape (*,*,K),
    data = extract_feature(data, feature_dim)
    # concatenate along the first dimension to an array of shape (*,K)
    data = data.reshape(data.shape[0] * data.shape[1], sliding_window_size)
    return data

#################################################
# Temp. main logic
#################################################

## ToDo: organize arch.
file_path = "/home/myriam/projects/emporus-ml-hw/data/2018-01-02.npy"
data = load_and_assert(file_path)

## ToDo: organize configuration
sliding_window_size = 30
sample_size = 10
feature_dim = 1
data = crop_and_reshape(data, sliding_window_size, sample_size,feature_dim)

print(data.shape)
