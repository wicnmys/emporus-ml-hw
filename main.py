import dataprocessing as dpr
import os
import numpy as np

#################################################
# Temp. main logic
#################################################

#ToDo: Organize arch.

#################################################
# Second part
#################################################

# 1. Load, normalize and concatenate the first 4 days as training data , sliding window k=360.
folder_path = "/home/myriam/projects/emporus-ml-hw/data"
training_data_filenames = ["2018-01-02.npy", "2018-01-03.npy", "2018-01-04.npy", "2018-01-05.npy"]
test_data_source = "2018-01-08.npy"


first = True
for filename in training_data_filenames:
    file_path = os.sep.join([folder_path, filename])
    data = dpr.load_data(file_path)
    if first:
        training_data = np.expand_dims(data, axis=0)
        first = False
    else:
        try:
            training_data = np.concatenate((training_data, np.expand_dims(data, axis=0)), axis=0)
        except ValueError:
            "data error: dimension mismatch along dates."

#2. Load and normalize the last day as test data, sliding window k=360.
file_path = os.sep.join([folder_path, test_data_source])
data = dpr.load_data(file_path)
test_data = np.expand_dims(data, axis=0)

print(test_data.shape)