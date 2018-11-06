from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from base import *


# func for decorate training model
@memory_usage
@calculation_time
def fit_model():
    model.fit(download_img('D:\TrainingSet\jpg', 'jpg', 49), categorical_res, batch_size=2, nb_epoch=100, verbose=1)


# results array creating
download_results = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
categorical_res = np_utils.to_categorical(download_results, 3)


# net structure
model = Sequential()
model.add(Dense(49, input_dim=49, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='softmax'))


# model compiling
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])


# model training
fit_model()


# model and weights saving
model_json = model.to_json()
with open('D:\TrainingSet\model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('D:\TrainingSet\model.h5')

