import pandas, numpy, pickle, keras, tensorflow, keras.backend.tensorflow_backend
from keras.layers import Dense
from keras.initializers import RandomNormal

# training argument
epochs = 100
validation_split = 0.01
gpu_fraction = 0.5

# load data
with open('data', 'rb') as f:
	x = numpy.float32(pickle.load(f))
with open('label', 'rb') as f:
	y = numpy.float32(pickle.load(f))
	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction))))

# set DNN model
def getmodel(factors):
	model = keras.models.Sequential()
	model.add(Dense(32, activation = 'linear', input_shape = (factors,), kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros'))
	model.add(Dense(16, activation = 'linear', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros'))
	model.add(Dense(1, activation = 'linear'))
	return model
			  
model = getmodel(len(x[0]))
model.compile(loss = 'mse', optimizer = 'adadelta')
model.summary()

earlystopping = keras.callbacks.EarlyStopping('val_loss', patience = 3, verbose = 1)
checkpoint = keras.callbacks.ModelCheckpoint("DNN_model", verbose = 1 ,save_best_only = True, monitor = 'val_loss')
history = model.fit(x, y, validation_split = validation_split, batch_size = 128, epochs = epochs, callbacks = [checkpoint, earlystopping])