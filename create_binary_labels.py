# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:10:33 2020

@author: Fafa
"""

import os
from scipy import signal
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
from scipy.integrate import simps
from sklearn import svm
import math

directory_string = '~\Desktop\project_code\DEAP\data_preprocessed_python'
directory = os.path.expanduser(directory_string)
filename='s0-.dat'

alpha_features=np.zeros((40,32,32))
theta_features=np.zeros((40,32,32))

flat_features=[]
valence_label=np.zeros((40,32))
arousal_label=np.zeros((40,32))
dominance_label=np.zeros((40,32))
liking_label=np.zeros((40,32))


sf=128
time=63
win=2*sf

for file in range(32):
    if file<=8:
        filename = filename.replace(filename[2],str(file+1))
        print(filename)
    else:
        filename = filename.replace(filename[1:3],str(file+1))
        print(filename)
        
    data_file_path = os.path.join(directory, filename)
    data_file = open(data_file_path, 'rb')
    pickle_file = cPickle.load(data_file, encoding='latin1')
    all_channels_data  = pickle_file['data']
    all_channels_data = all_channels_data[:,0:32,:]# we extract just eeg channels
    all_labels_values  = pickle_file['labels']
    for video in range(40):
        if all_labels_values[video,0]>=4.5:
            valence_label[video,file]=1
        else: 
            valence_label[video,file]=0
            
        if all_labels_values[video,1]>=4.5:
            arousal_label[video,file]=1
        else: 
            arousal_label[video,file]=0           
            
        if all_labels_values[video,2]>=4.5:
            dominance_label[video,file]=1
        else: 
            dominance_label[video,file]=0

        if all_labels_values[video,3]>=4.5:
            liking_label[video,file]=1
        else: 
            liking_label[video,file]=0             
        
        for channel in range(32):
            data=(all_channels_data[video,channel,384:])
            base_data=(all_channels_data[video,channel,:384])
            freqs, psd = signal.welch(data, sf, nperseg=win)
            base_freqs, base_psd = signal.welch(base_data, sf, nperseg=win)
            alpha_low, alpha_high= 8, 13
            idx_alpha= np.logical_and(freqs>=alpha_low, freqs<= alpha_high)
            base_idx_alpha= np.logical_and(base_freqs>=alpha_low, base_freqs<= alpha_high)
            # Frequency resolution
            freq_res = freqs[1] - freqs[0]
            base_freq_res = base_freqs[1] - base_freqs[0]
            alpha_power = simps(psd[idx_alpha], dx=freq_res)
            base_alpha_power= simps(base_psd[base_idx_alpha], dx=base_freq_res)
            alpha_power_decibel=10*math.log10(alpha_power/base_alpha_power)
            alpha_features[video,channel,file]=alpha_power_decibel
            print('Absolute alpha power: %.3f uV^2' % alpha_power +'decibel:%.3f' %alpha_power_decibel + 
            'for video: %.0f' %(video+1) + 
                  'and channel: %.0f' %(channel+1)+ "for participant: %.0f" %(file+1))
            
            
            theta_low, theta_high= 4, 8
            idx_theta= np.logical_and(freqs>=theta_low, freqs<= theta_high)
            base_idx_theta= np.logical_and(base_freqs>=theta_low, base_freqs<= theta_high)
            # Frequency resolution
            #theta_freq_res = freqs[1] - freqs[0]
            #base_freq_res = base_freqs[1] - base_freqs[0]
            theta_power = simps(psd[idx_theta], dx=freq_res)
            base_theta_power= simps(base_psd[base_idx_theta], dx=base_freq_res)
            theta_power_decibel=10*math.log10(theta_power/base_theta_power)
            theta_features[video,channel,file]=theta_power_decibel
            print('Absolute theta power: %.3f uV^2' % theta_power +'decibel:%.3f' %theta_power_decibel + 
            'for video: %.0f' %(video+1) + 
                  'and channel: %.0f' %(channel+1)+ "for participant: %.0f" %(file+1))
            
            
            

flat_valence_label=valence_label.flatten(order='F')
flat_arousal_label=arousal_label.flatten(order='F')
flat_dominance_label=dominance_label.flatten(order='F')
flat_liking_label=liking_label.flatten(order='F')



flat_valence_label=np.reshape(flat_valence_label,(flat_valence_label.shape[0],1))
flat_arousal_label=np.reshape(flat_arousal_label,(flat_arousal_label.shape[0],1))
                              
flat_alpha_features = np.transpose(alpha_features, (0,2,1))
flat_alpha_features = np.reshape(flat_alpha_features, (40*32, 32), order='F')

flat_theta_features = np.transpose(theta_features, (0,2,1)) 
flat_theta_features = np.reshape(flat_theta_features, (40*32, 32), order='F')

all_feature=np.concatenate((flat_alpha_features, flat_theta_features), axis=1)

"""
valence_label= valence_label.reshape(40*32,)
classifier=svm.SVC(gamma=0.001)
classifier.fit(features,valence_label)
predicted = classifier.predict(features)
t=0
for f in range(40):
    print(str(predicted[f])+' is ' + str(valence_label[f]))
    if predicted[f]==valence_label[f]:
        t=t+1

"""















                             