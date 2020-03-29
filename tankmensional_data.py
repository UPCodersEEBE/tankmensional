# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:46:09 2020

@author: crull
"""

import pandas as pd
import seaborn as sns
import sklearn
import numpy
import  pymongo
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pathlib
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

#Constants

ro=1000
densitat=ro
viscositat=0.00102

client=MongoClient('mongodb+srv://Alex:Alex@tankmensional-v6eso.mongodb.net/test?retryWrites=true&w=majority')
db = client['tankmension']
collection = db['dades']


#crear la base de dades de nou

pd.options.display.max_columns = None

df = pd.DataFrame(list(collection.find({})))


#df=pd.read_csv("db_rowdy.csv", sep=";")


df.rename(columns={'V (rpm)': 'velocitat',"Power (W)":"power"},inplace=True)

##df=df[df["Rodet"]=="Helix"]

df[["Rodet Diameter", "Bandes","velocitat", "power"]]=df[["Rodet Diameter", "Bandes","velocitat", "power"]].astype("float64")

df["velocitat"]=df["velocitat"]/60

sns.scatterplot(x=df["velocitat"], y=df["power"])

df["Re"]=ro*df["velocitat"]*df["Rodet Diameter"]/viscositat

df["Np"]=(df["power"]/(df["Rodet Diameter"]**5*df["velocitat"]**3*ro))



df["logRe"]=numpy.log(df["Re"])
df["logNp"]=numpy.log(df["Np"])

dataset = df

df=df[df["Series"]!="A1.1"]
df=df[df["velocitat"]<15]


for rodet in df["Series"].unique():
    df_especific=df[df["Series"]==rodet]
    lm = LinearRegression()
    X = df_especific[["logRe"]]
    Y = df_especific['logNp']
    lm.fit(X,Y)
    Yhat=lm.predict(X)
    Nphat=numpy.exp(Yhat)
    p=df["power"]
    phat=Nphat*df_especific["Rodet Diameter"]**5*df_especific["velocitat"]**3*ro
    RMSE=mean_squared_error(df_especific["power"], phat, squared=False)
    MSE=mean_squared_error(df_especific["power"], phat)
    sns.scatterplot(x=p,y=phat)
    plt.title(rodet)
    print(rodet, "RSME",round(RMSE,2))
    X = df_especific[["velocitat"]]
    Y = df_especific['power']
    lm.fit(X,Y)
    Yhat=lm.predict(X)
    RMSE=mean_squared_error(Y, Yhat, squared=False)
    MSE=mean_squared_error(Y, Yhat)
    print(rodet, "RSME",round(RMSE,2))
    print(rodet, round(RMSE,2))
    


lm = LinearRegression()

X = df[["logRe"]]
Y = df['logNp']

lm.fit(X,Y)
Yhat=lm.predict(X)
MSE=mean_squared_error(Y, Yhat)
RMSE=mean_squared_error(Y, Yhat, squared=False)

Nphat=numpy.exp(Yhat)

p=df["power"]
phat=Nphat*df["Rodet Diameter"]**5*df["velocitat"]**3*ro

df[["Circular", 'Disc', 'Helix', "Turbina"]] = pd.get_dummies(df["Rodet"])
##########################
####Tensorflow
dataset.pop("Series")
train_dataset = dataset.sample(frac=0.80,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("power")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('power')
test_labels = test_dataset.pop('power')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

EPOCHS = 996

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  )

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Power]')
plt.ylabel('Predictions [Power]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])



