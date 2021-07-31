import dataprocessing as dpr
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from knnmethod import FaissKNN, SklearnKNN

#################################################
# Temp. main logic
#################################################

#ToDo: Organize arch.

#################################################
# Second part
#################################################

# 1. Load, normalize and concatenate the first 4 days as training data , sliding window k=360.
## EDit the folder_path to point to the data folder
folder_path = "/home/myriam/projects/emporus-ml-hw/data"
training_data_filenames = ["2018-01-02.npy", "2018-01-03.npy", "2018-01-04.npy", "2018-01-05.npy"]
test_data_source = "2018-01-08.npy"

max_runtime = 30

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

k_factor = int(8*math.sqrt(test_data.shape[0]))

methods = [SklearnKNN('brute'), FaissKNN(360, "IVF" + str(k_factor) + ",Flat"),
           FaissKNN(360, "IVF" + str(k_factor) + ",PQ18")]

for method in methods:
    method.train(training_data)

speed = np.full((len(methods)+1, 256), np.inf)
similarity = np.full((len(methods)+1, 256), np.inf)

for n in range(1, 256+1):
    if np.sum(speed[0, np.isfinite(speed[0])]) < max_runtime:
        counter = 0
        for method in methods:
            toc = time.perf_counter()
            distances, indices = method.search(test_data, n)
            tic = time.perf_counter()
            speed[counter, n-1] = tic-toc
            similarity[counter, n-1] = np.mean(distances)
            counter += 1


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


axs[1,0].plot(np.arange(1, 100+1), similarity[0, :100], 'b')
axs[1,0].set_ylabel('distance')
axs[1,1].plot(np.arange(1, 100+1), similarity[1, :100], 'g')
axs[1,2].plot(np.arange(1, 100+1), similarity[2, :100], 'r')
axs[1,3].plot(np.arange(1, 100+1), similarity[0, :100], 'b',  np.arange(1, 100+1),
              similarity[1, :100], 'g', np.arange(1, 100+1), similarity[2, :100], 'r')

plt.show(block=True)
