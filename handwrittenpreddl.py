# -*- coding: utf-8 -*-
"""HandWrittenPredDL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oh4t-44gjPDpmvZFlglup3jtybpBEWiZ
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train.shape

x_test.shape

x_train=x_train.reshape(-1,28*28).astype('float64')/255
x_test=x_test.reshape(-1,28*28).astype('float64')/255

model=keras.Sequential(
    [
        keras.Input(shape=(784)),
        layers.Dense(512,activation='relu'),
        layers.Dense(128,activation='relu'),
        layers.Dense(10)
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train,epochs=5,batch_size=32,verbose=2)

model.evaluate(x_test,y_test,batch_size=32,verbose=2)

model.summary()
y_predicted=model.predict(x_test)

y_predicted_labels=[np.argmax(i) for i in y_predicted ]

cm=tf.math.confusion_matrix(
    y_test,
    y_predicted_labels
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Original')

prediction=model.predict(x_test)

prediction[0]

y_predicted_labels[0]

y_test[0]
