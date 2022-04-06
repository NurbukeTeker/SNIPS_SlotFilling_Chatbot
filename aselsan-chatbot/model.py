# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:44:52 2022

@author: nurbuketeker
"""
from model_train import show_predictions, model_func
import tensorflow as tf
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import LabelEncoder


device_name = tf.test.gpu_device_name()

device = torch.device("cpu")
print('GPU:', torch.cuda.get_device_name(0))

from transformers import AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

labelencoder = LabelEncoder()

global control 
control = 1
         
         
def getModel():
    global control 

    if control == 1 :
        # new model train
        tokenizer, joint_model, le, index_to_word = model_func()
        control = 0 
        return tokenizer, joint_model, le, index_to_word


tokenizer, joint_model, le, index_to_word = getModel()


# from keras.models import load_model

# joint_model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')
# import keras
# joint_model.save('path_to_my_model',save_format='tf')

# # Recreate the exact same model purely from the file
# new_model = keras.models.load_model('path_to_my_model')

# import pickle
# pickle.dump(joint_model, open("model_saved", 'wb'))
 
# # some time later...
 
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))


# def getPrediction(text):
#     result = show_predictions(text, tokenizer, joint_model, le, index_to_word)
#     return result


