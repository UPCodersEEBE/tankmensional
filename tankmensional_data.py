# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:46:09 2020

@author: crull
"""

import pandas as pd
import seaborn as sns
import sklearn
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient

#Constants

ro=1000
densitat=ro
viscositat=0.00102
mu=viscositat

#connectar a mongodb

client=MongoClient('mongodb+srv://Alex:Alex@tankmensional-v6eso.mongodb.net/test?retryWrites=true&w=majority')
db = client['tankmension']
collection = db['dades']


#crear la base de dades de nou

pd.options.display.max_columns = None

df = pd.DataFrame(list(collection.find({})))


df.rename(columns={'V (rpm)': 'velocitat',"Power (W)":"power"},inplace=True)

##df=df[df["Rodet"]=="Helix"]

a=[]
b=[]
c=[]
d=[]

for i in range(0,322):
    a.append('')
    b.append('')
    c.append('')
    d.append('')

df['Re']=a
df['Np']=b
df
df['logRe']=c
df['logNp']=d

for i in range(0,322):
    df["velocitat"][i]=float(df["velocitat"][i])
    df["Rodet Diameter"][i]=float(df["Rodet Diameter"][i])
    df["velocitat"][i]=float(df["velocitat"][i])
    df["power"][i]=float(df["power"][i])
    df["velocitat"][i]=df["velocitat"][i]/60

for i in range(0,322):    
    df["Re"][i]=ro*df["velocitat"][i]*df["Rodet Diameter"][i]/viscositat
    df["Np"][i]=(df["power"][i])/(df["Rodet Diameter"][i]**5*df["velocitat"][i]**3*ro)
    df["logRe"][i]=numpy.log(int(df["Re"][i]))
    df["logNp"][i]=numpy.log(int(df["Np"][i]))

sns.scatterplot(x=df["velocitat"], y=df["power"])


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



