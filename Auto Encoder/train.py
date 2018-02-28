import sys, csv, numpy, pandas, tensorflow, pickle, keras.backend.tensorflow_backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate
from sklearn.cluster import KMeans
from keras.initializers import RandomNormal


# training argument
epochs = 100
gpu_fraction = 1.0
encoders_dimension = [784, 128, 64, 32]
	
	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction))))


# set the autoencoder model
layer_encoders = []
layer_decoders = []
for i in range(1, len(encoders_dimension)):      
	encoder = Dense(encoders_dimension[i], activation = 'relu', input_shape = (encoders_dimension[i-1],), kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')
	layer_encoders.append(encoder)
	
	decoder = Dense(encoders_dimension[i-1], activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')
	layer_decoders.append(decoder)

encoder = Sequential(layer_encoders)
layer_decoders.reverse()
autoencoder = Sequential(layer_encoders + layer_decoders)
autoencoder.compile(loss = 'mse', optimizer = 'adadelta')
print(autoencoder.summary())


# training
if sys.argv[1] != sys.argv[1]:
	input_data = numpy.load("image.npy")	
	autoencoder.fit(input_data, input_data, epochs = epochs)
	with open('model', 'wb') as f:
		pickle.dump(KMeans(2, n_init = 200).fit(encoder.predict(input_data, verbose = 1)).labels_, f)


# testing
if sys.argv[1] == sys.argv[1]:
	# load testing data
	with open('model', 'rb') as f:
		cluster = pickle.load(f)

	input_data = pandas.read_csv(sys.argv[2], sep = ',', encoding = 'utf-8', usecols = ['image1_index', 'image2_index'])
	input_data2 = input_data['image2_index'].values
	input_data = input_data['image1_index'].values

	with open(sys.argv[3], 'w', encoding='utf-8') as f:
		spamwriter = csv.writer(f, delimiter=',')
		spamwriter.writerow(['ID', 'Ans'])
		for i, j in enumerate(input_data):  
			if cluster[j] - cluster[input_data2[i]] == 0:
				spamwriter.writerow([str(i), str(1)])
			else:
				spamwriter.writerow([str(i), str(0)])