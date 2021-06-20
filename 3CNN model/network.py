# -*- coding: utf-8 -*-
"""network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1REX29BUu4DhKIlJc3rLu8_QYpFL8Dver
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import metrics

input_set_labeled = np.genfromtxt ('training.csv', delimiter=",")
input_set=input_set_labeled[1:input_set_labeled.shape[0],:]
input_data=input_set[:,1:input_set.shape[1]]
target_data = input_set[:,0]
target_data=target_data.reshape(target_data.shape[0],1)
print(input_data.shape)
print(target_data.shape)
print(input_set.shape[1])
print(input_set.shape[0])

sess = tf.Session()
init_var = tf.global_variables_initializer()
sess.run(init_var)
indices1 = tf.cast(target_data, tf.int32)
one_hot_vecs1 = tf.one_hot(indices1,np.max(sess.run(indices1))+1)
new_output=sess.run(one_hot_vecs1)
new_output=new_output.reshape(21000,10)
print(new_output)

input_train,input_test, target_train, target_test =train_test_split(input_data,new_output,test_size=0.166667, random_state=10)
print(input_train.shape)
print(input_test.shape)
print(target_train.shape)
print(target_test.shape)

input_train_map=input_train.reshape(-1,28,28,1)
input_test_map=input_test.reshape(-1,28,28,1)
print(input_train_map.shape)
print(input_test_map.shape)

score=[]
kf = KFold(n_splits=5)
index=0
for train_index, test_index in kf.split(input_train_map):

    X_train, X_test_loop = input_train_map[train_index], input_train_map[test_index]
    y_train, y_test_loop = target_train[train_index], target_train[test_index]
    print(X_train.shape)
    model = Sequential()
    #32 core，with size 3*3
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    #2*2 max pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #64 core，with size 3*3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #2*2 max pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #flatten layor
    model.add(Flatten())
    #full connected
    model.add(Dense(512, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    #sorting
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    filepath=str(index)+'weights.best.hdf5'
    index=index+1
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True,
                            mode='auto')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train, batch_size=96, epochs=100,callbacks=callbacks_list,verbose=0)
#     model.fit(X_train, y_train,  batch_size=147, epochs=1000,callbacks=callbacks_list,verbose=0)
    score.append(model.evaluate(X_test_loop, y_test_loop, batch_size=147))

print("score=: ",score)
print("average score in five loop: ",np.mean(score))
final_score=[]
for iteration in range (0,5):
  filepath=str(iteration)+'weights.best.hdf5'
#   print(filepath)
  model.load_weights(filepath)
  model.save(str(iteration)+"CNN.model")
  final_score.append(model.evaluate(input_test_map, target_test, batch_size=147))
# final_score=model.evaluate(X_test, y_test, batch_size=147)
print(final_score)
print("best score=: " ,np.min(final_score))
# print(X_validation)