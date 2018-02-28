import os,re
import numpy,_pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from itertools import islice 


class DataManager:
	def __init__(self):
		self.data={}

	def add_data(self,name,data_path,with_label=True):
		print('read data from %s...'%data_path)
		X,Y=[],[]
		with open(data_path,'r') as f:
			for line in f:
				if with_label:
					lines=line.strip().split(' +++$+++ ')	
					Y.append(int(lines[0]))
					lines=re.sub("([^a-zA-Z0-9] |[^a-zA-Z0-9 ])","",lines[1])
					X.append(lines)
				else:
					lines=re.sub("([^a-zA-Z0-9] |[^a-zA-Z0-9 ])","",line)
					X.append(lines)

		if with_label:
			self.data[name]=[X,Y]
		else:
			self.data[name]=[X]
			
	def add_data2(self,data_path):
		print('read data from %s...'%data_path)
		X,Y=[],[]
		with open(data_path,'r') as f:
			for line in islice(f, 1, None):  
				lines=re.split("^[\d]+,", line.strip())[1]
				lines=re.sub("([^a-zA-Z0-9] |[^a-zA-Z0-9 ])","",lines)
				X.append(lines)
		
		self.data['semi_data']=self.data['semi_data'][0]
		self.data['semi_data']=self.data['semi_data']+X
		self.data['semi_data']=[self.data['semi_data']]
			
	def add_test_data(self,name,data_path):
		print('read data from %s...'%data_path)
		X,Y=[],[]
		with open(data_path,'r') as f:
			for line in islice(f, 1, None):  
				lines=re.split("^[\d]+,", line.strip())[1]
				lines=re.sub("([^a-zA-Z0-9] |[^a-zA-Z0-9 ])","",lines)
				X.append(lines)

		self.data[name]=[X]

	def tokenize(self,vocab_size):
		print('create new tokenizer')
		self.tokenizer=Tokenizer(num_words=vocab_size)
		for key in self.data:
			print('tokenizing %s'%key)
			texts=self.data[key][0]
			self.tokenizer.fit_on_texts(texts)

	def save_tokenizer(self,path):
		print('save tokenizer to %s'%path)
		_pickle.dump(self.tokenizer,open(path,'wb'))

	def load_tokenizer(self,path):
		print('Load tokenizer from %s'%path)
		self.tokenizer=_pickle.load(open(path,'rb'))

	def to_sequence(self,maxlen):
		self.maxlen=maxlen
		for key in self.data:
			print('Converting %s tosequences'%key)
			tmp=self.tokenizer.texts_to_sequences(self.data[key][0])
			self.data[key][0]=numpy.array(pad_sequences(tmp,maxlen=maxlen))

	def to_bow(self):
		for key in self.data:
			print('Converting %s totfidf'%key)
			self.data[key][0]=self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')

	def to_category(self):
		for key in self.data:
			if len(self.data[key])==2:
				self.data[key][1]=numpy.array(to_categorical(self.data[key][1]))

	def get_semi_data(self,name,label,threshold,loss_function):
		label=numpy.squeeze(label)
		index=(label>1-threshold)+(label<threshold)
		semi_X=self.data[name][0]
		semi_Y=numpy.greater(label,0.5).astype(numpy.int32)
		if loss_function=='binary_crossentropy':
			return semi_X[index,:],semi_Y[index]
		elif loss_function=='categorical_crossentropy':
			return semi_X[index,:],to_categorical(semi_Y[index])
		else:
			raiseException('Unknown loss function:%s'%loss_function)

	def get_data(self,name):
		return self.data[name]

	def split_data(self,name,ratio):
		data=self.data[name]
		X=data[0]
		Y=data[1]
		data_size=len(X)
		val_size=int(data_size*ratio)
		return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
