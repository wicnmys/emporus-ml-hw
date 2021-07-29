import dataprocessing as dpr
import os
import numpy as np
import math

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
        training_data = data
        first = False
    else:
        try:
            training_data = np.concatenate((training_data, data))
        except ValueError:
            "data error: dimension mismatch along dates."

#2. Load and normalize the last day as test data, sliding window k=360.
file_path = os.sep.join([folder_path, test_data_source])
test_data = dpr.load_data(file_path)

# 3. Perform a knn search on the train data for the test data:
k=10
# a. Once using SKlearn-nearestneighbors module, any base algorithm.

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(training_data)
distances, indices = nbrs.kneighbors(test_data[:5])
print(distances)
print(indices)
# REMARKS from https://scikit-learn.org/stable/modules/neighbors.html
# when D > 15, the intrinsic dimensionality of the data is generally
# too high for tree-based methods ==> BRUTE

# b. Once using FAISS-library (use a flat index).
import faiss
index = faiss.IndexFlatL2(360)
index.add(training_data)
distances, indices = index.search(test_data[:5], k)
print(distances)
print(indices)

# c. Once using FAISS-library (use any non flat index).
# with the help of https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
# A. How big is the dataset? below 1M vectors => IVF K ,where K is 4*sqrt(N) to 16*sqrt(N)
k_factor = int(8*math.sqrt(test_data.shape[0]))
# B. Is memory a concern? Quite important = > OPQM_D,...,PQMx4fsr
index = faiss.index_factory(360, "IVF" + k_factor +",PQ16")
index.add(training_data)
distances, indices = index.search(test_data[:5], k)
print(distances)
print(indices)


## Why are the distances equal to 0?
a = test_data[0,:]
print(a)
b = training_data[indices[0,0], :]
print(b)
dist = np.linalg.norm(a-b)
print(dist)



