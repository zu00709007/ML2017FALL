import pandas, numpy, pickle, keras, tensorflow, keras.backend.tensorflow_backend
from keras.layers import Input, Embedding, Flatten, Dot, Add

# training argument
epochs = 137
validation_split = 0.01
gpu_fraction = 0.5

# load data
with open('data', 'rb') as f:
	x = numpy.float32(pickle.load(f))
with open('label', 'rb') as f:
	y = numpy.float32(pickle.load(f))
	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction))))

# set linear model
def getmodel():
	X = Input(shape = [9])
	Y = Input(shape = [1])
	
	W = Embedding(1, 9)(Y)
	W = Flatten()(W)
	
	bias = Embedding(1, 1)(Y)
	bias = Flatten()(bias)
	
	model = Dot(axes = 1)([X, W])
	model = Add()([model, bias])
	model = keras.models.Model([X, Y], model)
	return model
			  
model = getmodel()
model.compile(loss = 'mse', optimizer = 'adadelta')
model.summary()

tmp = numpy.array([0] * len(x))
earlystopping = keras.callbacks.EarlyStopping('val_loss', patience = 3, verbose = 1)
checkpoint = keras.callbacks.ModelCheckpoint("gradient_descent_model", verbose = 1 ,save_best_only = True, monitor = 'val_loss')
history = model.fit([x, tmp], y, validation_split = validation_split, batch_size = 128, epochs = epochs, callbacks = [checkpoint, earlystopping])