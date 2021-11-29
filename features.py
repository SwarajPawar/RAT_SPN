
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

plt.figure(figsize=[5,5])

# Display the first image in training data
#plt.subplot(121)
#plt.imshow(train_X[0,:,:], cmap='gray')
#plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
#plt.subplot(122)
#plt.imshow(test_X[0,:,:], cmap='gray')
#plt.title("Ground Truth : {}".format(test_Y[0]))


# Data pre processing $$WHY ARE WE DOING THIS
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
#train_X.shape, test_X.shape


#Converting data fromat from int8 to float32 AND rescaling pixel values to 0-1
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
#print('Original label:', train_Y[0])
#print('After conversion to one-hot:', train_Y_one_hot[0])


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


#Adding dropout layer to avoid overfitting
batch_size = 64
epochs = 200
num_classes = 10

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

digits_model.add(Dense(64, activation='linear'))
digits_model.add(LeakyReLU(alpha=0.1))           
digits_model.add(Dropout(0.3))
digits_model.add(Dense(num_classes, activation='softmax'))

digits_model.summary()


print(train_X.shape)
print(train_Y)

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
	#wr.writerow(columns)
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
	#wr.writerow(columns)
	for row in test_data:
			wr.writerow(row)

'''
1) l [                    ]
2) l [                    ]
3) l 
.  l
.
.
.
.
.




'''
'''
digits_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
digits_model_dropout = digits_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label),callbacks=ModelCheckpoint("model_{acc}.hdf5"))
digits_model.save("digits_model_dropout.h5py")

#Model evaluation on the test set
test_eval = digits_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


#plot the accuracy and loss plots between training and validation data:
accuracy = digits_model_dropout.history['acc']
val_accuracy = digits_model_dropout.history['val_acc']
loss = digits_model_dropout.history['loss']
val_loss = digits_model_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#Predict labels 
predicted_classes = digits_model.predict(test_X)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape
correct = np.where(predicted_classes==test_Y)[0]
print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_Y)[0]
print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()


#Saving predicted labels of classes testX
pint(predicted_classes)

#Classification report 
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


#Confusion matrix 
from sklearn.metrics import confusion_matrix
#matrix = metrics.confusion_matrix(test_Y.argmax(axis=1), y_pred.argmax(axis=1))
confusion_matrix(test_X,test_Y)

'''
