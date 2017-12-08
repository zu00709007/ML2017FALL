import argparse,os,csv
import numpy,tensorflow
import keras.backend.tensorflow_backend

from util import DataManager
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Model,Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,Dropout,Bidirectional
from keras.callbacks import EarlyStopping,ModelCheckpoint

parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['train','test'])
parser.add_argument('first')
parser.add_argument('second')
args=parser.parse_args()

# training argument
epochs=5
batch_size=128
validation=0.1
loss_function='binary_crossentropy'
hidden_size=512
dropout_rate=0
threshold=0.1
gpu_fraction=0.3
vocab_size=20000
max_length=40
embedding_dim=128

save_path='./'
load_path='./'
	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction))))

print('initial model...')
model = Sequential()	
model.add(Embedding(vocab_size,embedding_dim,trainable=True,input_shape=(max_length,)))

# RNN layer
model.add(Bidirectional(LSTM(hidden_size,dropout=dropout_rate)))
model.add(Dense(1,activation='sigmoid'))

# other setting
adam=Adam()
model.compile(loss=loss_function,optimizer=adam,metrics=['accuracy'])

dm=DataManager()
if args.action=='train':
	dm.add_data('train_data',args.first,True)
	dm.add_data('semi_data',args.second,False)
	dm.tokenize(vocab_size)
	dm.save_tokenizer(os.path.join(save_path,'token.pk'))
	dm.to_sequence(max_length)
	print('start supervised learning')
	(X,Y),(X_val,Y_val)=dm.split_data('train_data',validation)
	earlystopping=EarlyStopping(monitor='val_acc',patience=5,verbose=1,mode='max')

	save_path=os.path.join(save_path,'model.h5')
	checkpoint=ModelCheckpoint(filepath=save_path,verbose=1,save_best_only=True,save_weights_only=True,monitor='val_acc',mode='max')
	history=model.fit(X,Y,validation_data=(X_val,Y_val),epochs=epochs,batch_size=batch_size,callbacks=[checkpoint,earlystopping])
	
	print('start semi-supervised learning')
	model.load_weights(save_path)
	[semi_all_X]=dm.get_data('semi_data')
	
	for i in range(30):
		semi_pred=model.predict(semi_all_X,batch_size=1024,verbose=True)
		semi_X,semi_Y=dm.get_semi_data('semi_data',semi_pred,threshold,loss_function)
		semi_X=numpy.concatenate((semi_X,X))
		semi_Y=numpy.concatenate((semi_Y,Y))
		print('iteration %d semi_datasize:%d'%(i+1,len(semi_X)))

		history=model.fit(semi_X,semi_Y,validation_data=(X_val,Y_val),epochs=2,batch_size=batch_size,callbacks=[checkpoint,earlystopping])

		if os.path.exists(save_path):
			print('load model from %s'%save_path)
			model.load_weights(save_path)
		else:
			raiseValueError("Can't find the file %s"%path)

elif args.action=='test':
	dm.add_test_data('test_data',args.first)
	dm.load_tokenizer(os.path.join(load_path,'token.pk'))
	dm.to_sequence(max_length)
	model.load_weights(os.path.join(load_path,'model.h5'))
	print('start predict')
	[test_all_X]=dm.get_data('test_data')
	with open(args.second, 'w', encoding='utf-8') as f:
		spamwriter = csv.writer(f, delimiter=',')
		spamwriter.writerow(['id', 'label'])
		test_predict=model.predict(test_all_X,batch_size=1024)
		tmp = len(test_predict)
		for i in range(tmp):
			spamwriter.writerow([str(i), str(numpy.greater(test_predict[i][0], 0.5).astype(numpy.int32))])
