# The relevant environment is varx

# Generate a simple vascular relationship network, then try to infer it back using vector autoregression

'''
import tifffile as tf
from scipy import io as sio
from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import zarr
from numcodecs import Blosc, Zstd 
'''
	
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
import zarr
import numpy as np
from numcodecs import Blosc, Zstd 
from scipy import stats
import matplotlib.pyplot as plt

### Build vascular relationship network
#       ___2___
#     1/       \ 3
#__0__/         \____4___
#     \         / 
#     5\___6___/ 7
#
num_time_samples = 3000#
num_vessels = 8 # start with a oversimplified network composed of 8 vessel segments and two bifurcations as depicted above. Blood flow in vessels 0,1,5 is measurable and is driven by blood flow in vessels 2,6 
sampling_frequency_hz = 10 # 30 volumes per second smoothed to 10 samples per second
rbc_velocity = 5 # start with a constant velocity of red blood cells across the entire network. RBC transit time and inter-vascular latency are dictated by vascular distance d (t=d/v_rbc), with no dispersion.

cbf = np.zeros((num_vessels,num_time_samples)) # each line is a vessel, each column is a step in time


'''
# contrived dataset with dependency
data = list()

for j in range(num_time_samples):
    v1 = np.divide(j , num_time_samples) + 1e-1*random() + np.sin(2*np.pi*sampling_frequency_hz*j)
    v2 = 0.5*v1 + 1e-1*random()
    row = [v1, v2]
    data.append(row)
'''

time_vector = np.linspace(0, num_time_samples*sampling_frequency_hz,num_time_samples)
v1 = np.sin(2*np.pi*sampling_frequency_hz*time_vector*0.03) + 2
v1 += 0.2* (1 + np.random.randn(*v1.shape))
print(v1)
print(stats.describe(v1))
lag_value = 20 * sampling_frequency_hz
v2 = 0.5*np.roll(v1,lag_value) + 1e-0*random()
cbf = np.transpose( np.stack((v1,v2))   )
plt.plot(time_vector,cbf)
plt.show()
used_cbf = cbf[lag_value:-lag_value,:]
forecast_time_vector = time_vector[-lag_value] + time_vector
short_time_vector = time_vector[lag_value:-lag_value]
print(cbf.shape)

data = list()
data.append(cbf)

# print(data)


# fit model
model = VAR(used_cbf)
model_fit = model.fit(maxlags=int(lag_value*1.8))
# make prediction
print(model_fit.summary())
yhat = model_fit.forecast(model_fit.y, steps=num_time_samples)
#print(yhat)
plt.plot(short_time_vector,used_cbf)
plt.plot(forecast_time_vector,yhat)
plt.show()



def save_dataset(save_file_name,group_name,dataset):
    root = zarr.open_group(save_file_name, mode='w')
    fill_me = root.require_group(group_name)
    root[group_name] = dataset

zarr.storage.default_compressor = Zstd(level=3)
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)


'''
vascular_lengths= 20 * np.random.rand(num_vessels)

#vascular_latency_vector = np.rint((vascular_lengths*sampling_frequency_hz / rbc_velocity), dtype='int64')
vascular_latency_vector = np.rint(vascular_lengths * sampling_frequency_hz * np.reciprocal(rbc_velocity)).astype('int64')

vascular_latency_matrix = np.zeros((num_vessels,num_vessels), dtype='int64')
vascular_coupling_matrix = np.zeros((num_vessels,num_vessels))

rng = np.random.default_rng()

vascular_latency_matrix[1,2] = vascular_latency_vector[2]
vascular_latency_matrix[5,6] = vascular_latency_vector[6]
vascular_latency_matrix[0,1] = vascular_latency_vector[1]
vascular_latency_matrix[0,5] = vascular_latency_vector[5]


vascular_coupling_matrix[1,2] = 0.9
vascular_coupling_matrix[5,6] = 0.8
vascular_coupling_matrix[0,1] = 0.4
vascular_coupling_matrix[0,5] = 0.6

cbf[2,:]= np.random.randn(1,num_time_samples) * rng.random() + (10*rng.random()) 
cbf[6,:]= np.random.randn(1,num_time_samples) * rng.random() + (10*rng.random())

cbf[1,:] = vascular_coupling_matrix[1,2] * np.roll(cbf[2,:], -vascular_latency_matrix[1,2])
cbf[5,:] = vascular_coupling_matrix[5,6] * np.roll(cbf[6,:], -vascular_latency_matrix[5,6])
c01 = vascular_coupling_matrix[0,1] * np.roll(cbf[1,:], -vascular_latency_matrix[0,1])
c05 = vascular_coupling_matrix[0,5] * np.roll(cbf[5,:], -vascular_latency_matrix[0, 5])
cbf[0,:] = c01 + c05

sos = signal.butter(4, 0.3, 'low', output='sos', fs=sampling_frequency_hz)
filtered = signal.sosfilt(sos, cbf, axis=1)

start_from = 100
t_vec = np.linspace(0,num_time_samples/sampling_frequency_hz, num_time_samples)
cbf = cbf.T
filtered = filtered.T
#plt.plot(t_vec[start_from:], cbf[start_from:,:])
plt.plot(t_vec[start_from:], filtered[start_from:,:])
plt.show()




print('all done')


'''