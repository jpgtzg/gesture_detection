# Written by Juan Pablo Gutierrez
import keras
import numpy as np
import tensorflow as tf
from keras import layers

model = keras.Sequential()

model.add(layers.Embedding(input_dim=1000, output_dim=64))

model.add(layers.GRU(128))

model.add(layers.Dense(10))

model.summary()
