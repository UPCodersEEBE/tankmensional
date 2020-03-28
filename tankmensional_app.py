# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:06:22 2020

@author: crull
"""


from flask import Flask, render_template
app = Flask(__name__)


@app.route("/")
def template_test():
    return render_template('sample.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])


if __name__ == '__main__':
    app.run()