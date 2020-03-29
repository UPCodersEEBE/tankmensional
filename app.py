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

@app.route("/analytics")
def analytics():
    import tankmensional_data as tkdata
    df=tkdata.df
    rodet_count=list(df.groupby("Rodet").count()["Series"])
    rodet=list(df.groupby("Rodet").count()["Series"].index.values)
    nom="nom"
    return render_template('analytics.html', project_id="HLT", rodet_count=rodet_count, rodet=rodet, nom="nom")


@app.route('/getdata', methods=["POST", "GET"])
def getdata():
    if request.method == "POST":
        collection.insert_one({'\ufeff_id': str(results.count()+1), "Series": request.form["series"], "Rodet": request.form["rodet"], "Rodet Diameter": request.form["rodetD"], "Bandes": request.form["bandes"], "V (rpm)": request.form["V"], "Power (W)": request.form["power"]})
        return redirect("/")
    else:
        return render_template("AddData.html")

if __name__ == '__main__':
    app.run()