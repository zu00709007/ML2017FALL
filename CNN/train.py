from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Sequential
import numpy
import pandas
import sys

# build model
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size=(5,5), padding = 'valid', input_shape = (48, 48, 1), activation='relu')) 
model.add(ZeroPadding2D(padding = (2, 2), data_format = 'channels_last'))
model.add(MaxPooling2D(pool_size = (5, 5), strides = (2, 2)))
model.add(ZeroPadding2D(padding = (1, 1), data_format = 'channels_last'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1), data_format = 'channels_last'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(AveragePooling2D(pool_size = (3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding = (1, 1), data_format = 'channels_last'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1), data_format = 'channels_last'))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1), data_format = 'channels_last'))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))
model.summary()

# read in data and split data
x_item = []
tmp = pandas.read_csv(sys.argv[1], usecols=[1]).values.tolist()
for i in tmp:
	x_item.append(i[0].split(' '))
x_item = numpy.array(x_item).reshape(-1,48,48,1)
tmp = pandas.read_csv(sys.argv[1], usecols=[0])
tmp = numpy.array(tmp)
y_item = numpy.zeros((len(tmp), 7))
for i in range(0, len(tmp)):
	y_item[i][tmp[i][0]] = 1

# set learning rate and crossentropy
opt = Adam(lr = 0.00001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x = x_item, y = y_item, epochs = 150, batch_size = 128)

# save the model
model.save('./model.h5') 