# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:06:22 2020

@author: crull
"""

from flask import Flask, render_template, request, redirect
from pymongo import MongoClient

alex=MongoClient('mongodb+srv://Alex:Alex@tankmensional-v6eso.mongodb.net/test?retryWrites=true&w=majority')
db = alex['tankmension']
collection = db['dades']


results=collection.find({})


app = Flask(__name__)


@app.route("/")
def template_test():
    return render_template('sample.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])


@app.route("/about")
def about():
    return render_template('about.html',my_string="Wheeeee!", my_list=[0,1,2,3,4,5])


@app.route('/getdata', methods=["POST", "GET"])
def getdata():
    if request.method == "POST":
        collection.insert_one({'\ufeff_id': str(results.count()+1), "Series": request.form["series"], "Rodet": request.form["rodet"], "Rodet Diameter": request.form["rodetD"], "Bandes": request.form["bandes"], "V (rpm)": request.form["V"], "Power (W)": request.form["power"]})
        return redirect("/")
    else:
        return render_template("AddData.html")
    
value = []
@app.route("/chooseplot", methods=["GET","POST"])
def ChoosePlot():
    types = ['PlotHelix', "PlotTurbina","PlotDisc","PlotCircular"]
    if request.method == 'POST':
        for i in types:
            post_id = request.form.get(i)
            if post_id is not None:
                value.append([post_id,request.form.get("velocity")])
                return redirect("/analytics")
    else:
        return render_template('ChooseRodet.html')


    
@app.route("/analytics")
def analytics():
    import tankmensional_data as tkdata
    import pandas as pd
    rodet=value[len(value)-1][0]
    linial_res=tkdata.linial_res
    x=linial_res[rodet][0]
    y=linial_res[rodet][1]
    velocity = value[len(value) - 1][1]
    Yhat=round(tkdata.linial.predict(pd.DataFrame([float(velocity)]))[0],1)
    return render_template('analytics.html', project_id=rodet, velocity=velocity, rodet=rodet, nom="nom", y=y, x=x, Yhat=Yhat)


@app.route("/one", methods=["GET", "POST"])
def second():
    return (render_template('charts2.html'))

app.run(debug=True, use_reloader=False)
