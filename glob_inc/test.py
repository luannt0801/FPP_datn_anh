import paho.mqtt.publish as publish
import numpy
import json
import zlib,sys
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from pandas import read_csv, concat

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss

import tldextract
import numpy as np
import tensorflow as tf
global_model = tf.keras.models.load_model("demo_work/simple_LSTM_model")
global_model.summary()
b = pickle.dumps(global_model.get_weights())
publish.single(topic="dynamicFL/model", payload=b, hostname="broker.emqx.io")