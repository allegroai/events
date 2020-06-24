from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
import tempfile
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from trains.storage import StorageManager
import numpy as np
from trains import Task
task = Task.init(project_name='webinar',task_name='keras mnist')


def get_data_trains():
    data_path = StorageManager.get_local_copy(
        remote_url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        name="mnist"
    )

    with np.load(data_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

#batch_size = 128
#num_classes = 10
#epochs = 12

params = {'batch_size': 128, 'num_classes' : 10, 'epochs' : 2}
task.connect(params)
task.add_tags(tags=["webinar","demo","fun"])

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = get_data_trains()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, params['num_classes'])
y_test = keras.utils.to_categorical(y_test, params['num_classes'])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(params['num_classes'], activation='softmax'))

output_folder = os.path.join(tempfile.gettempdir(), 'keras_example')
board = TensorBoard(histogram_freq=1, log_dir=output_folder, write_images=True, )
model_save = ModelCheckpoint(
    filepath=os.path.join(output_folder, 'weight.{epoch}.hdf5'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=params['batch_size'],
          epochs=params['epochs'],
          verbose=1,
          validation_data=(x_test, y_test) ,
          callbacks=[board,model_save])
score = model.evaluate(x_test, y_test, verbose=0, callbacks=[board])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
logger = task.get_logger()
logger.report_scalar(title="results",series='loss',iteration=1, value=score[0])
logger.report_scalar(title='results',series='accuracy',iteration=1, value=score[1])