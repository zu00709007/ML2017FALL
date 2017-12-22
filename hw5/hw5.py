import sys, csv, tensorflow, pandas, keras.backend.tensorflow_backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Input, Flatten, Dot, Add


# training argument
epochs = 30
validation_split = 0.1
dropout_rate = 0.1
gpu_fraction = 0.5
dimension = 120

	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction))))


# load training data
#input_data = pandas.read_csv("train.csv", sep = ',', encoding='utf-8', usecols=['UserID', 'MovieID', 'Rating'])
#max_user = input_data['UserID'].drop_duplicates().max()
#max_movie = input_data['MovieID'].drop_duplicates().max()
#input_data = input_data.sample(frac = 1., random_state = 168464)
#users = input_data['UserID'].values - 1
#movies = input_data['MovieID'].values - 1
#ratings = input_data['Rating'].values
max_user = 6040
max_movie = 3952
	
	
# set the UV decomposition model
print('initial model...')
def getmodel(users, items, factors, dropout_rate):
	P_ = Input(shape = [1])
	P = Embedding(users, factors)(P_)
	P = Flatten()(P)
	
	Q_ = Input(shape = [1])
	Q = Embedding(items, factors)(Q_)
	Q = Flatten()(Q)
	
	P_b = Embedding(users, 1)(P_)
	P_b = Flatten()(P_b)
	
	Q_b = Embedding(items, 1)(Q_)
	Q_b = Flatten()(Q_b)
	
	model = Dot(axes = 1)([P, Q])
	model - Add()([model, P_b, Q_b])
	model = keras.models.Model([P_, Q_], model)
	return model
			  
model = getmodel(max_user, max_movie, dimension, dropout_rate)
model.compile(loss = 'mse', optimizer = 'adamax')
model.summary()


# training
if sys.argv[1] == "train":
	print('start to train data')	
	earlystopping = EarlyStopping('val_loss', patience = 3, verbose = 1)
	checkpoint = ModelCheckpoint("normal_model", verbose = 1 ,save_best_only = True, monitor = 'val_loss')
	history = model.fit([users, movies], ratings, validation_split = validation_split, batch_size = 128, epochs = epochs, callbacks=[checkpoint,earlystopping])

	
# testing
if sys.argv[1] == "test":
	# load testing data
	print('start to test data')
	input_data = pandas.read_csv(sys.argv[2], sep = ',', encoding='utf-8', usecols=['UserID', 'MovieID'])
	users = input_data['UserID'].values - 1
	movies = input_data['MovieID'].values - 1
	model.load_weights('normal_model')
	output = model.predict([users, movies])
	with open(sys.argv[3], 'w', encoding='utf-8') as f:
		spamwriter = csv.writer(f, delimiter=',')
		spamwriter.writerow(['TestDataID', 'Rating'])
		for i, j in enumerate(output):  
			spamwriter.writerow([str(i+1), str(j[0])])