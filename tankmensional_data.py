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
#import pathlib
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


linial_res={}

for rodet in df["Rodet"].unique():
    df_especific=df[df["Rodet"]==rodet]
    linial = LinearRegression()
    X = df_especific[["velocitat"]]
    Y = df_especific['power']
    linial.fit(X,Y)
    Yhat=linial.predict(X)
    RMSE=mean_squared_error(Y, Yhat, squared=False)
    MSE=mean_squared_error(Y, Yhat)
    line_pred=linial.predict(pd.DataFrame(range(100,1200,100)))
    linial_res[rodet]=[list(range(100,1200,100)),list(line_pred), RMSE]
    
for rodet in df["Rodet"].unique():
    df_especific=df[df["Rodet"]==rodet]
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

#df[["Circular", 'Disc', 'Helix', "Turbina"]] = pd.get_dummies(df["Rodet"])



