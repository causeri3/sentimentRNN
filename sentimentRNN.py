#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:58:21 2020

@author: Vanessa Causemann
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#set directory
os.chdir('/Users/vanessa/Desktop/Programming-Stuff/RNN')

#import social media data,the example used has ~ 100.000 [observations] x 4 [one column for comments, 3 for positive, negative, neutral dummies/one-hot]
#smileys in comments where translated into made up words as 'cryandlaughsmiley'

df = pd.read_csv ('data3.csv', sep=';')

df=df.dropna()
df=df.iloc[:,:].to_numpy()

#####TRANSLATE SOCIAL MEDIA COMMENTS INTO ENGLISH#####################################################################################################################################################################################

from googletrans import Translator

translator = Translator()

results = list(range(len(df[:,0]))) 

#google is smart and has a limit
for i in range(len(df[:,0])):
    results[i] = translator.translate(df[i,0], dest="en").text
    print(results[i])
    print(i)
    
#thus, run it in batches of 10 with 5 sec pause in between
import time
    
for i in range(len(df[:,0])):
    if i % 10 != 0:
        results[i] = translator.translate(df[i,0], dest="en").text
        print(results[i])
        print(i)
    else:
        time.sleep(5)
        results[i] = translator.translate(df[i,0], dest="en").text
        print(results[i])
        print(i)
        
        
#check if translation worked (trial and error for the batch size and seconds in loop above)
results[436]
translator.translate(df[436,0], dest="en").text
results[298]
translator.translate(df[298,0], dest="en").text
results[197]
translator.translate(df[197,0], dest="en").text

  
#convert and back-up translation
results_vec = np.array([results]).transpose()
np.save('trans.npy', results_vec)

#####TOKENIZE ENGLISH COMMENTS########################################################################################################################################################################################################

trans=np.load('trans.npy')
from keras.preprocessing.text import Tokenizer

#make words to tokens, save index-dictionary and save as X=training examples
t=Tokenizer(lower = False, split = ' ')
t.fit_on_texts(trans[:,0].tolist())
idx_word = t.index_word
sequences = t.texts_to_sequences(trans[:,0].tolist())
X=np.array([sequences]).transpose()

#check if retokenized words fit with translated comments
print('Retokenized:', ' '.join(idx_word[w] for w in sequences[2]))
print('OriginalTrans:', trans[2,0])
' '.join(idx_word[w] for w in sequences[2])
print('Retokenized:',' '.join(idx_word[w] for w in X[12345,0]))
print('OriginalTrans:',trans[12345,0])

#see index-dictionary and check for random words
print(t.word_index)
idx_word[23701]


#####PREPARE MATRICES FOR RNN#########################################################################################################################################################################################################
#save labels (dummies/one-hot) as Y=output
Y=np.array(df[:,1:4], dtype = int)

#padding with 0s to get matrix

l = max(len(x) for sublist in X for x in sublist)

pad=X.tolist()
X=np.zeros((len(X), l), dtype = int)

for i in range(len(X)):
    for j in range (len(np.array(pad[i], dtype = int)[0,:])):
        X[i,j]= (np.array(pad[i], dtype = int)[0,j])

#get observations and labels back together for shuffling
data = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]

#shuffle observations and split into training, validation and test sets 60/20/20
np.random.seed(9668)
np.random.shuffle(data)

X_train = data[1:round(len(data)*0.6),0:1123]
Y_train = data[1:round(len(data)*0.6),1123:1126]
X_valid = data[round(len(data)*0.6):round(len(data)*0.8),0:1123]
Y_valid = data[round(len(data)*0.6):round(len(data)*0.8),1123:1126]
X_test = data[round(len(data)*0.8):len(data),0:1123]
Y_test = data[round(len(data)*0.8):len(data),1123:1126]

#####TRAIN MODEL######################################################################################################################################################################################################################

#set-up the architecture of the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(idx_word)+1, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(Y_train.shape[1], activation='softmax')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#train 
history = model.fit(X_train,Y_train, epochs=3,
                    validation_data=(X_valid, Y_valid))


#get your metrics set up
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
  
#check your metrics  
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plot_graphs(history, 'recall')
plot_graphs(history, 'precision')

#get predictions
predictions = model.predict(X_test)
predictions[predictions>=0.5] = 1
predictions[predictions<0.5] = 0

#spot check predictions
print(predictions[1:50,:])
Y_test[1:50,:]

#####TUNE MODEL#######################################################################################################################################################################################################################
#weighting of classes improved model very efficiently
#in this case the labelled data was biased towards the neutral class (human stamina)
#therefore the metric recall is a good choice to choose epochs

class_weight = {0: 75.,
                1: 1.,
                2: 25.}

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(idx_word)+1, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(Y_train.shape[1], activation='softmax')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

#Model1
history = model.fit(X_train,Y_train, epochs=16,
                    validation_data=(X_valid, Y_valid), class_weight=class_weight)

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
  
  
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plot_graphs(history, 'recall')
plot_graphs(history, 'precision')

#back-up and get tuned model 'Model1'
model.save('Model1')
Model1 = tf.keras.models.load_model('Model1')

#get predictions to compute confusion-matrix

predModel1 = Model1.predict(data[:,0:1123])

predModel1[predModel1>=0.5] = 1
predModel1[predModel1<0.5] = 0

predModel1 = predModel1.astype(int)

predictionsModel1 = predModel1

label_pred = predictionsModel1[:,0]
label_pred[predictionsModel1[:,2]==1] = 2

labels = data[:,1123]
labels[data[:,1125]==1]=2

confusion =  pd.DataFrame(tf.math.confusion_matrix(labels = labels, predictions =label_pred).numpy())

#####EXPORT PREDICTED LABELS FOR MANUAL CHECKING######################################################################################################################################################################################

#predict over unshuffled data
data_unshuffled = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]

predModel1 = Model1.predict(data_unshuffled[:,0:1123])

predModel1[predModel1>=0.5] = 1
predModel1[predModel1<0.5] = 0

predModel1 = predModel1.astype(int)

#retoken comments to words
blubb = list(range(len(data_unshuffled))) 
retoken = []

for i in range(len(data)):
    blubb[i] = data_unshuffled[i,0:1123][data_unshuffled[i,0:1123] != 0].tolist()
    retoken.append(' '.join(idx_word[w] for w in blubb[i]))


d = {'comments': retoken, 'positiv': data_unshuffled[:,1123], 'neutral':data_unshuffled[:,1124], 'negativ':data_unshuffled[:,1125], 'pred_positive': predModel1[:,0], 'pred_neutral':predModel1[:,1], 'pred_negativ':predModel1[:,2]}
                  
Model1Results = pd.DataFrame(d)
Model1Results.to_csv('Model1Results.csv', encoding='utf-8', index=False)

d1 = {'translation': trans.tolist()}
Translation = pd.DataFrame(d1)
Translation.to_csv('Translation.csv', encoding='utf-8', index=False)


