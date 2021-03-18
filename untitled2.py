# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:08:25 2020

@author: Fafa
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot
import tensorflow as tf
from sklearn.metrics import accuracy_score

from untitled0 import flat_valence_label, flat_alpha_features, flat_arousal_label, all_feature
#w=tf.Variable([0],dtype=tf.float32)
#x=tf.placeholder(tf.float32, [3,1])
#%%
X_train=all_feature[120:,:]
X_test=all_feature[:120,:]
y_train=flat_valence_label[120:,:]
y_test=flat_valence_label[:120,:]

"""

model= tf.keras.models.Sequential([
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='C:\\temp\\tensorflow_logs', histogram_freq=1)

model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard_callback])
model.summary()



predictions = model.predict_classes(X_test)
print("Accuracy = "+ str(accuracy_score(y_test,predictions)))
"""

#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        fmodel= tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        fmodel.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

        fmodel.fit(X_train[:,chromosome],y_train, epochs=50)
        predictions = fmodel.predict_classes(X_train[:,chromosome])
        scores.append(accuracy_score(y_train,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score


chromo,score=generations(size=20,n_feat=64,n_parents=5,mutation_rate=0.10,
                     n_gen=100,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
#%%
model= tf.keras.models.Sequential([
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='C:\\temp\\tensorflow_logs', histogram_freq=1)

model.fit(X_train[:,chromo[-1]], y_train, epochs=50, callbacks=[tensorboard_callback])
model.summary()

predictions = model.predict_classes(X_test[:,chromo[-1]])
print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))

"""
model.fit(X_train[:,chromo[-1]],y_train)
predictions = model.predict(X_test[:,chromo[-1]])
print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))
model.evaluate(all_feature[:120,:], flat_valence_label[:120,:])
"""
