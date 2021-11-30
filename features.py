
	#Edges : [2773, 12551, 128239, 128239, 128239, 128239, 180126, 180126, 180126, 180126, 180126]
	#Layers : [20, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20]
	
	
from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

#Analyzing data, remove this section later
import numpy as np
from tensorflow.keras.utils import to_categorical     #Changed
import matplotlib.pyplot as plt

print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
#print('Total number of outputs : ', nClasses)
#print('Output classes : ', classes)


# Data pre processing $$WHY ARE WE DOING THIS
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
#train_X.shape, test_X.shape


#Converting data fromat from int8 to float32 AND rescaling pixel values to 0-1
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.


#model The Data
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import 
from keras.layers import LayerNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow import keras
import csv



digits_model = Sequential()
digits_model.add(Conv2D(4, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
digits_model.add(LeakyReLU(alpha=0.1))
digits_model.add(MaxPooling2D((2, 2),padding='same'))
digits_model.add(Dropout(0.25))
digits_model.add(Conv2D(8, (3, 3), activation='linear',padding='same'))
digits_model.add(LeakyReLU(alpha=0.1))
digits_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
digits_model.add(Dropout(0.25))
digits_model.add(Conv2D(16, (3, 3), activation='linear',padding='same'))
digits_model.add(LeakyReLU(alpha=0.1))                  
digits_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
digits_model.add(Dropout(0.4))
digits_model.add(Conv2D(32, (3, 3), activation='linear',padding='same'))
digits_model.add(LeakyReLU(alpha=0.1))                  
digits_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
digits_model.add(Dropout(0.4))
digits_model.add(Flatten())

digits_model.summary()



train_data = list()

for i, img in enumerate(train_X):
	if (i+1)%100==0:
		print(i+1)
	img = np.expand_dims(img, axis=0)
	fv = digits_model.predict(img)[0]
	instance = [i+1, train_Y[i]] + list(fv)
	train_data.append(instance)


with open('ftrain.tsv', 'w') as file:
	wr = csv.writer(file,delimiter='\t')
	for row in train_data:
			wr.writerow(row)


test_data = list()

for i, img in enumerate(test_X):
	if (i+1)%100==0:
		print(i+1)
	img = np.expand_dims(img, axis=0)
	fv = digits_model.predict(img)[0]
	instance = [i+1, test_Y[i]] + list(fv)
	test_data.append(instance)


with open('ftest.tsv', 'w') as file:
	wr = csv.writer(file,delimiter='\t')
	for row in test_data:
			wr.writerow(row)
