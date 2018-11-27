from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils
import timeit

start = timeit.default_timer()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10) # 10 classes

model = Sequential()

model.add(Conv2D( filters=32, kernel_size=(3,3), data_format="channels_last",
                  activation='relu',
                  input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(Dropout(0.3))
#model.add(Conv2D(filters=64, kernel_size=(2,2)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#model.add(Dropout(0.3))
#model.add(Conv2D(filters=128, kernel_size=(3,3)))
#model.add(Dropout(0.3))
model.add(Flatten())
#model.add(Dense(128, activation='softmax'))
model.add(Dense(10, activation='softmax')) # classifies 10 classes

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=256, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("mnist.cnn.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("mnist.cnn.h5")
print("Saved model to disk")

print('Done.{0:f}'.format(timeit.default_timer() - start))