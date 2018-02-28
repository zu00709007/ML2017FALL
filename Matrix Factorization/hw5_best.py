import sys, csv, numpy, tensorflow, pandas, keras.backend.tensorflow_backend
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate


# training argument
epochs = 100
validation_split = 0.1
dropout_rate = 0.2
gpu_fraction = 0.5
dimension = 180

	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction))))


# load training data
#input_data = pandas.read_csv("train.csv", sep = ',', encoding='utf-8', usecols=['UserID', 'MovieID', 'Rating'])
#max_user = input_data['UserID'].drop_duplicates().max()
#max_movie = input_data['MovieID'].drop_duplicates().max()
#input_data = input_data.sample(frac = 1., random_state = 25415)
#users = input_data['UserID'].values - 1
#movies = input_data['MovieID'].values - 1
#ratings = input_data['Rating'].values
#input_data = pandas.read_csv(sys.argv[5], delimiter = '::', encoding = 'utf-8', usecols = ['UserID', 'Gender', 'Age'], engine = 'python')
#input_data = input_data.sort_values(by = ['UserID']).values
#gender = []
#age = []
#for i in users:
#	if input_data[i][1] == 'M':
#		gender.append(1)
#	else:
#		gender.append(0)
#	age.append(input_data[i][2] // 5)
#gender = numpy.array(gender)
#age = numpy.array(age)
#max_age = numpy.amax(age) + 1
max_age = 12
max_user = 6040
max_movie = 3952
	
	
# set the UV decomposition model
print('initial model')
def getmodel(users, items, age, dimension, dropout_rate):
	P_ = Input(shape = [1])
	P = Embedding(users, dimension)(P_)
	P = Flatten()(P)
	
	Q_ = Input(shape = [1])
	Q = Embedding(items, dimension)(Q_)
	Q = Flatten()(Q)
	
	R_ = Input(shape = [1])
	R = Embedding(2, dimension)(R_)
	R = Flatten()(R)
	
	S_ = Input(shape = [1])
	S = Embedding(age, dimension)(S_)
	S = Flatten()(S)
	
	model = Concatenate()([P, Q])
	model = Concatenate()([model, R])
	model = Concatenate()([model, S])
	model = Dropout(dropout_rate)(model)
	model = Dense(dimension, activation = 'relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(1, activation = 'linear')(model)
	model = keras.models.Model([P_, Q_, R_, S_], model)
	return model
			  
model = getmodel(max_user, max_movie, max_age, dimension, dropout_rate)
model.compile(loss = 'mse', optimizer = 'adamax')
model.summary()


# training
if sys.argv[1] == "train":
	print('start to train data')	
	earlystopping = EarlyStopping('val_loss', patience = 3, verbose = 1)
	checkpoint = ModelCheckpoint("best_model", verbose = 1 ,save_best_only = True, monitor = 'val_loss')
	history = model.fit([users, movies, gender, age], ratings, validation_split = validation_split, batch_size = 128, epochs = epochs, callbacks = [checkpoint, earlystopping])

	
# testing
if sys.argv[1] == "test":
	# load testing data
	print('start to test data')
	input_data = pandas.read_csv(sys.argv[2], sep = ',', encoding = 'utf-8', usecols = ['UserID', 'MovieID'])
	users = input_data['UserID'].values - 1
	movies = input_data['MovieID'].values - 1
	input_data = pandas.read_csv(sys.argv[5], delimiter = '::', encoding = 'utf-8', usecols = ['UserID', 'Gender', 'Age'], engine = 'python')
	input_data = input_data.sort_values(by=['UserID']).values
	gender = []
	age = []
	for i in users:
		if input_data[i][1] == 'M':
			gender.append(1)
		else:
			gender.append(0)
		age.append(input_data[i][2] // 5)
	gender = numpy.array(gender)
	age = numpy.array(age)
	max_age = numpy.amax(age) + 1
	model.load_weights('best_model')
	output = model.predict([users, movies, gender, age], verbose = 1)
	with open(sys.argv[3], 'w', encoding='utf-8') as f:
		spamwriter = csv.writer(f, delimiter=',')
		spamwriter.writerow(['TestDataID', 'Rating'])
		for i, j in enumerate(output):  
			spamwriter.writerow([str(i+1), str(j[0])])