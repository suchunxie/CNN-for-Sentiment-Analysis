
import tensorflow as tf
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from pickle import load
from pickle import dump
import string
from numpy import array
#from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
#from tensorflow import keras.model
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Concatenate

#use the last 100 positive reviews 
#and the last 100 negative reviews as a test set (100 reviews) and
# the remaining 1,800 reviews as the training dataset.

#This is a 90% train, 10% split of the data.
#reviews named 000 to 899 are for training data 
#and reviews named 900 onwards are for test.

# 000-899是训练集， 900-999是测试集

# load doc into memory 
def load_doc(filename):
    file = open(filename, "r") # open file as read only
    text = file.read()
    file.close()
    return text


# turn a doc into clean tokens
#  词形还原功能
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens



# load all docs in a directory
def process_docs(directory, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc)
		# add to list
		documents.append(tokens)
	return documents


# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)


	# load all training reviews
negative_docs = process_docs('D:/Data/operate_Html/Text processing/txt_sentoken/neg', True)
#print(negative_docs[0:2])

positive_docs = process_docs('D:/Data/operate_Html/Text processing/txt_sentoken/pos', True)
trainX = negative_docs + positive_docs
trainy = [0 for _ in range(900)] + [1 for _ in range(900)]
save_dataset([trainX, trainy], 'train.pkl')
 
# load all test reviews
negative_docs = process_docs('D:/Data/operate_Html/Text processing/txt_sentoken/neg', False)
positive_docs = process_docs('D:/Data/operate_Html/Text processing/txt_sentoken/pos', False)
testX = negative_docs + positive_docs
print("negative_docs[0] {}".format(negative_docs[0]))
testY = [0 for _ in range(100)] + [1 for _ in range(100)]
save_dataset([testX, testY], 'test.pkl')

# -----------------------
#-------------------------
#---------------------------


# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))



# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

import string
# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])



# 直接用tokenizer来建 vocabulary

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# define the model
# Function API used, not Sequential , 
# Function API is more Flexible, 
# from tensorflow import keras
# from tensorflow.keras import layers

# https://tensorflow.google.cn/guide/keras/functional?hl=zh-cn

# change kernel size in the convolution layer
def define_model(length, vocab_size):
	# channel 1
	inputs1= tf.keras.Input(shape=(length,))
	embedding1= Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size =4, activation="relu")(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1= Flatten()(pool1)
	# channel2
	inputs2 = tf.keras.Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2= Conv1D(filters= 32, kernel_size = 6, activation="relu")(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size= 2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = tf.keras.Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size = 8, activation ="relu")(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3= Flatten()(pool3)
	# merge
	merged = Concatenate()([flat1, flat2, flat3])	
	# interpretation
	dense1= Dense(10, activation="relu")(merged)
	outputs =  Dense(1, activation ="sigmoid")(dense1)
	model = Model(inputs= [inputs1, inputs2, inputs3], outputs = outputs)

	#compile
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

	# summarize
	print(model.summary())
	plot_model(model, show_shapes= True, to_file="multichannel.png")
	return model

#load training dataset
trainLines, trainLabels = load_dataset("train.pkl")
tokenizer = create_tokenizer(trainLines)
#calculate max docs length
length = max_length(trainLines)
vocab_size = len(tokenizer.word_index) +1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
print(trainX.shape)

# define model
model = define_model(length, vocab_size)
model.fit([trainX, trainX, trainX], array(trainLabels), epochs =7, batch_size = 16)
# save the model
model.save("model.h5")


from pickle import load
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Evaluate
# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')
 
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)
 
# load the model
model = load_model('model.h5')
 
model.fit([trainX, trainX, trainX], array(trainLabels), epochs =7, batch_size = 16, verbose = 0)

# evaluate model on training dataset
loss, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %f' % (acc*100))
 
# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX,testX,testX], array(testLabels), verbose=0)
print('Test Accuracy: %f' % (acc*100))

model.predict([trainX,trainX,trainX])
#print("training score {}".format(model.score([trainX,trainX,trainX], array(testLabels))))


# Input layer that defines the length of input sequences.
# Embedding layer set to the size of the vocabulary and 100-dimensional real-valued representations.
# One-dimensional convolutional layer with 32 filters and a kernel size set to the number of words to read at once.
# Max Pooling layer to consolidate the output from the convolutional layer.
# Flatten layer to reduce the three-dimensional output to two dimensional for concatenation.








