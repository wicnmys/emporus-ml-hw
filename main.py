import numpy as np

def load_file(file_path):
    data = np.load(file_path)
    assert (len(data.shape) == 3) & (data.shape[2] == 7)
    return data

file_path = "/home/myriam/projects/emporus-ml-hw/data/2018-01-02.npy"
data = load_file(file_path)

