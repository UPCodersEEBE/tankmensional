# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:46:09 2020

@author: crull
"""


import pandas as pd
import seaborn as sns
import sklearn

#Constants

ro=1000
densitat=ro
viscositat=0.00102

df=pd.read_csv("db_rowdy.csv", sep=";")

df.rename(columns={'V (rpm)': 'velocitat',"Power (W)":"power"},inplace=True)

df["velocitat"]=df["velocitat"]/60

sns.scatterplot(x=df["velocitat"], y=df["power"])

df["Re"]=ro*df["velocitat"]*df["Rodet Diameter"]/viscositat

df["Np"]=(df["power"]/(df["Rodet Diameter"]**5*df["velocitat"]**3*ro))







