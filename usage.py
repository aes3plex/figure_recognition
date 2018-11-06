from keras.engine.saving import model_from_json

from base import *


# func for decorate training model
@memory_usage
@calculation_time
def the_prediction():
    global prediction
    prediction = loaded_model.predict(test_img('D:\TrainingSet\square_5.jpg'))


# loading and compiling model
with open('D:\TrainingSet\model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("D:\TrainingSet\model.h5")

loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])


# testing model
the_prediction()
print('Figure is: ' + switch(np.argmax(prediction)))





