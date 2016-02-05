"""
Module name :   model_io
Usage       :   from speech_io import model_io
Description :   read/write HDF5/pickle format models
                write log files

Author:         zhashi@microsoft.com
Last update:    2015/12
"""

import cPickle
import time

import keras.models
from keras.models import model_from_json


####################################################################################
def pickle(file, data):
    print "Saving file: ", file
    fo = open(file, 'wb')
    cPickle.dump(data, fo, protocol = 2)
    fo.close()

def unpickle(file):
    print "Loading file: ", file
    fo = open(file, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic

####################################################################################
"""
 In Keras, It is not recommended to use pickle or cPickle to save a Keras model. For example,
    cPickle.dump(model,open("./model.pkl","wb"))
 is not recommended.
"""

def save_keras_model(model, model_architecture_filename='my_model_architecture.json', model_weight_filename='my_model_weights.h5'):

    # save as JSON, only save the architecture of a model, and not its weights
    json_string = model.to_json()
    with open(model_architecture_filename, 'w') as json_f:
        json_f.write(json_string)

    # save the weights only
    model.save_weights(model_weight_filename, overwrite=True)

def load_keras_model(model_architecture_filename='my_model_architecture.json', model_weight_filename='my_model_weights.h5'):

    # model reconstruction from JSON:
    with open(model_architecture_filename, 'r') as json_f:
        model = model_from_json(json_f.read())

    # Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the same architecture:
    model.load_weights(model_weight_filename)

    # Note: remember to call model.compile. This can be done either before or after the model.load_weights call but must be after the model architecture is specified and before the model.predict call.
    # for example,
    # model.compile(loss='mse', optimizer=sgd)
    return model

####################################################################################

def save_la_model(model, filename):
    import lasagne
    data = lasagne.layers.get_all_param_values(model)
    pickle(filename, data)

def load_la_model(model, filename):
    import lasagne
    data = unpickle(filename)
    lasagne.layers.set_all_param_values(model, data)

####################################################################################

class Logger():
    def __init__(self, save_file = "default_log.txt"):
        self.save_file = save_file
        self.logf = open(save_file, "a+")
        timestamp = time.strftime("%m/%d/%Y %H:%M:%S\n", time.localtime(time.time()))
        self.logf.write("LOG START\n")
        self.logf.write(timestamp + '\n\n')
        self.alive = True

    def log(self, string):
        assert self.alive
        if string[-1]!= "\n":
            string += "\n"
        self.logf.write(string)
        self.logf.flush()

    def log_and_print(self, string):
        print string
        self.log(string)

    def close(self):
        if self.alive == True:
            self.logf.write("\n"*2)
            self.logf.write("LOG END\n")
            timestamp = time.strftime("%m/%d/%Y %H:%M:%S\n", time.localtime(time.time()))
            self.logf.write(timestamp+'\n\n')
            self.logf.close()
            self.alive = False

    def __del__(self):
        self.close()

####################################################################################
def tuple_multiply(tuple_to_multiply):
    result = 1

    for tuple_item in tuple_to_multiply:
        result = result * tuple_item

    return result