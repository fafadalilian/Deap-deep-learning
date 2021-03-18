# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:19:44 2020

@author: Fafa
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from create_labels import flat_valence_label, flat_alpha_features, flat_arousal_label, all_feature
#w=tf.Variable([0],dtype=tf.float32)
#x=tf.placeholder(tf.float32, [3,1])
#%%
model= tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

#from tensorflow.keras.optimizers im
model.compile(optimizer='RMSprop',
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='C:\\temp\\tensorflow_logs', histogram_freq=1)

model.fit(all_feature[120:,:], flat_valence_label[120:,:], epochs=300, callbacks=[tensorboard_callback])
model.summary()

Y_hat= model.predict(all_feature[:120,:])

model.evaluate(all_feature[:120,:], flat_valence_label[:120,:])
