import dataprocessing as dpr
import os
import numpy as np
import math
import time
import matplotlib as plt

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

max_runtime_per_method = 120

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
sklearn_nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(training_data)
# REMARKS from https://scikit-learn.org/stable/modules/neighbors.html
# when D > 15, the intrinsic dimensionality of the data is generally
# too high for tree-based methods ==> BRUTE

# b. Once using FAISS-library (use a flat index).
import faiss
faiss_bsc = faiss.IndexFlatL2(360)
faiss_bsc.add(training_data)

# c. Once using FAISS-library (use any non flat index).
# with the help of https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
# A. How big is the dataset? below 1M vectors => IVF K ,where K is 4*sqrt(N) to 16*sqrt(N)
k_factor = int(8*math.sqrt(test_data.shape[0]))
# B. Is memory a concern? Quite important = > OPQM_D,...,PQMx4fsr
# 4 <= M <= 64
# MUST: d % M == 0
factory_string = "IVF" + str(k_factor) + ",PQ18"
faiss_sop = faiss.index_factory(360, factory_string)
assert not faiss_sop.is_trained
faiss_sop.train(training_data)
assert faiss_sop.is_trained
faiss_sop.add(training_data)

speed = np.full((3, 256), np.inf)
# using Euclidean, see
# https://medium.com/@gshriya195/top-5-distance-similarity-measures-implementation-in-machine-learning-1f68b9ecb0a3
similarity = np.full((3, 256), np.inf)


for n in range(1, 256+1):
    ## sklearn
    if np.sum(speed[0, np.isfinite(speed[0])]) < max_runtime_per_method:
        toc = time.perf_counter()
        distances, indices = sklearn_nbrs.kneighbors(test_data, n_neighbors=n)
        tic = time.perf_counter()
        speed[0, n-1] = tic-to  c
        similarity[0, n-1] = np.mean(distances)

    ## faiss flat
    if np.sum(speed[1, np.isfinite(speed[1])]) < max_runtime_per_method:
        toc = time.perf_counter()
        distances, indices = faiss_bsc.search(test_data, n)
        tic = time.perf_counter()
        speed[1, n-1] = tic-toc
        similarity[1, n - 1] = np.mean(distances)

    ## faiss non-flat
    if np.sum(speed[2, np.isfinite(speed[2])]) < max_runtime_per_method:
        toc = time.perf_counter()
        distances, indices = faiss_sop.search(test_data, n)
        tic = time.perf_counter()
        speed[2, n-1] = tic-toc
        similarity[2, n - 1] = np.mean(distances)

##################################
# Plots
#################################


fig, axs = plt.subplots(2, 4)
axs[0,0].plot(np.arange(1, 256+1), speed[0], 'b')
axs[0,0].set_ylabel('seconds')
axs[0,0].set_title('sklearn')
axs[0,1].plot(np.arange(1, 256+1), speed[1], 'g')
axs[0,1].set_title('faiss-flat')
axs[0,2].plot(np.arange(1, 256+1), speed[2], 'r')
axs[0,2].set_title('faiss-not-flat')
axs[0,3].plot(np.arange(1, 256+1), speed[0], 'b',  np.arange(1, 256+1),
                    speed[1], 'g', np.arange(1, 256+1), speed[2], 'r')
axs[0,3].set_title('all')


axs[1,0].plot(np.arange(1, 256+1), similarity[0], 'b')
axs[1,0].set_ylabel('distance')
axs[1,1].plot(np.arange(1, 256+1), similarity[1], 'g')
axs[1,2].plot(np.arange(1, 256+1), similarity[2], 'r')
axs[1,3].plot(np.arange(1, 256+1), similarity[0], 'b',  np.arange(1, 256+1),
              similarity[1], 'g', np.arange(1, 256+1), similarity[2], 'r')
