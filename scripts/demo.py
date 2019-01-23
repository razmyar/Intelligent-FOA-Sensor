from __future__ import print_function
import glob
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('th')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


plt.style.use('ggplot')
os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

root_path = "dataset/"
cat_path = root_path + "cat/"
model_path = root_path + "model/weights.best.hdf5"

# 2 for original, 3 for smooth
ACC_DATA_COL = 2
ROUND_DECIMAL = 2

# train and test parameters
train_ratio = 0.8
eval_ratio = 0.25
BATCH_SIZE = 60
NUM_EPOCH = 150
dataset_ratio = 1  # Use the entire dataset

# Read ACC data
img_label = []
img_path = []
for path, subdirs, files in os.walk(cat_path):
    for name in files:
        img_list = (os.path.join(path, name))
        img_label.append(img_list.split("/cat/")[1].split("/")[0])
        img_path.append(img_list)


img_path = pd.DataFrame(img_path)
img_label = pd.DataFrame(img_label)
full_data = pd.concat([img_path, img_label], axis=1)
full_data.columns = ['image', 'label']
full_data = full_data[1:]

# Shuffle data
full_data = full_data.sample(frac=1).reset_index(drop=True)


data_labels = full_data['label']
data_labels = data_labels.as_matrix()

# Number of unique labels
unique_labels = np.unique(data_labels)
NUM_LABELS = len(unique_labels)
print("Number of unique labels:", NUM_LABELS)
print(unique_labels, "\n")

# Converts labels to category
unique_labels = list(unique_labels)

data_labels = list(data_labels)
unique_labels.sort()
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
idx_to_label = {i: lbl for i, lbl in enumerate(unique_labels)}
data_labels = list(label_to_idx[data] for data in data_labels)

ratio = np.zeros(NUM_LABELS)
for data in data_labels:
    ratio[data] += 1

for i in range(0, len(ratio)):
    print("Label", idx_to_label[i], " num member:", ratio[i], "ratio:",
          float(ratio[i] / len(data_labels)) * 100)

data_labels = np.asanyarray(data_labels).astype(np.int64)

data_labels_one_hot = np_utils.to_categorical(data_labels, NUM_LABELS)


# Function to read imgs
def read_img(filelist):
    return np.array([np.array(Image.open(fname).convert('L')) for fname in filelist])


# Read img files
file_list = full_data['image']
data_image = read_img(file_list)


IMAGE_HEIGHT = data_image.shape[1]
IMAGE_WIDTH = data_image.shape[2]
DATASET_SIZE = data_image.shape[0]


if len(data_image.shape) == 4:
    NUM_CHANNEL = data_image.shape[3]
else:
    NUM_CHANNEL = 1


data_image = data_image.reshape(DATASET_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
                                NUM_CHANNEL)

# convert data type to integer
data_image = data_image.astype('float32')


row = (train_ratio * data_image.shape[0])

# Test Set, train set
X_train = data_image[:int(row), ...]
Y_train = data_labels_one_hot[:int(row)]

X_test = data_image[int(row):, ...]
Y_test = data_labels_one_hot[int(row):]

row = (eval_ratio * X_train.shape[0])
X_eval = X_train[:int(row), ...]
Y_eval = Y_train[:int(row)]

print('\n# X_train: ', X_train.shape[0])
print('# Y_train: ', Y_train.shape[0])

print('# X_test: ', X_test.shape[0])
print('# Y_test: ', Y_test.shape[0])

print('# X_eval: ', X_eval.shape[0])
print('# X_eval: ', X_eval.shape[0])



K.set_image_data_format('channels_last')
model = Sequential()

model.add(Conv2D(32,
                 kernel_size=(10, 10),
                 activation='relu',
                 input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNEL)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64,
                 kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNEL)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNEL)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.75))


model.add(Dense(NUM_LABELS, activation='softmax'))

print (model.summary())

# load model is exist
if os.path.exists(model_path):
    # os.system('clear')
    print("\nLoading the model...\n")
    model.load_weights(model_path)
else:
    # os.system('clear')
    print("\nNo model found to load.\n")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# save Model
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
# TensorBoard
vis = TensorBoard(log_dir=root_path + "log",
                  histogram_freq=0,
                  write_graph=True,
                  write_images=False)

callbacks_list = [checkpoint, vis]

# create an image generator
train_datagen = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=40,
                                   zca_whitening=False,
                                   data_format=K.set_image_data_format(
                                       'channels_last')
                                   )

# fit parameters from data
# Computes any statistical required to actually perform the transfer
train_datagen.fit(X_train)

eval_datagen = ImageDataGenerator(rescale=1.0)
eval_datagen.fit(X_eval)

test_datagen = ImageDataGenerator(rescale=1.0)
test_datagen.fit(X_test)

history = model.fit_generator(train_datagen.flow(X_train, Y_train,
                                                 batch_size=BATCH_SIZE),
                              validation_data=eval_datagen.flow(
                                  X_eval, Y_eval),
                              validation_steps=len(X_eval) / BATCH_SIZE,
                              steps_per_epoch=len(X_eval) / BATCH_SIZE,
                              callbacks=callbacks_list,
                              epochs=NUM_EPOCH)



score = model.evaluate(X_test, Y_test, verbose=1)
print("\rTest loss", score[0]),
print("\rTest accuracy:", score[1] * 100, "%")

from sklearn.metrics import classification_report

predicted_y = model.predict(X_test)
predicted_y = np.argmax(predicted_y, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print("result:\n\n")

Y_test = list(idx_to_label[data] for data in Y_test)
predicted_y = list(idx_to_label[data] for data in predicted_y)

Y_test = [elem for elem in Y_test]
predicted_y = [elem for elem in predicted_y]

# for i in range(0,len(Y_test)):
#  print ("  Y_test: ", Y_test[i], "Predicted Y: ", predicted_y[i])


print(classification_report(Y_test, predicted_y))
