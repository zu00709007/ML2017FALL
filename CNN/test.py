from keras.models import Model, load_model
import numpy
import pandas
import csv
import sys

# read in data and split data
x_item = []
tmp = pandas.read_csv(sys.argv[1], usecols=[1]).values.tolist()
for i in tmp:
	x_item.append(i[0].split( ))
x_item = numpy.array(x_item)
x_item = x_item.reshape(-1,48,48,1)

# load model 
model = load_model('./best_model.h5')
model.summary()

# start predict
tmp = len(x_item)
with open(sys.argv[2], 'w', encoding='utf-8') as f:
	spamwriter = csv.writer(f, delimiter=',')
	spamwriter.writerow(['id', 'label'])
	# start testing x_vector to get y
	for i in range(tmp):
		spamwriter.writerow([str(i), str(numpy.argmax(model.predict(x_item[i:i+1]), axis=1)[0])])