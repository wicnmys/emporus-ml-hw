import dataprocessing as dpr

#################################################
# Temp. main logic
#################################################

## ToDo: organize arch.
file_path = "/home/myriam/projects/emporus-ml-hw/data/2018-01-02.npy"
data = dpr.load_and_assert(file_path)

## ToDo: organize configuration
sliding_window_size = 30
sample_size = 10
feature_dim = 1

data = dpr.crop_and_reshape(data, sliding_window_size, sample_size, feature_dim)
data = dpr.normalize_samples(data)

print(data.shape)
data[36, 5] = float('NaN')
data[243, 5] = float('Inf')
data = dpr.throw_nan_inf(data)
print(data.shape)