import numpy as np


#################################################
# First part
#################################################

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


def normalize_samples(data, epsilon=1e-100):
    # 3. Normalize each sample (first dimension) of the new array. The normalized array should
    # have, for every i: {mean(arr[i,:]) == 0, std(arr[i,:]) == 1}
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + epsilon)


def throw_nan_inf(data):
    # 4. Throw away any sample that has any NaN or inf elements
    log = np.all(np.isfinite(data), axis=1)
    return data[log, :]


def load_data(file_path, sliding_window_size=360, sample_size=10, feature_dim=1):
    data = load_and_assert(file_path)
    data = crop_and_reshape(data, sliding_window_size, sample_size, feature_dim)
    # HANDLE  AttributeError: 'float' object has no attribute 'sqrt' from
    # normalize_samples function
    # solution from: https://www.programmersought.com/article/21818799868/
    data = np.float64(data)
    data = normalize_samples(data)
    data = throw_nan_inf(data)
    # HANDLE  TypeError: in method 'IndexFlat_add', argument 3 of type 'float const *'
    # from faiss.index.add(training_data)
    # solution from: https://github.com/facebookresearch/faiss/issues/461
    data = np.float32(data)
    return data
