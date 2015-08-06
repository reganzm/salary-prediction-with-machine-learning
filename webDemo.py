#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
import requests

import Analyser as A
clf = A.get_clf()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        return render_template('predict.html',jd=request.form['jd'])
    if request.method == 'GET':
        return render_template('predict.html')

@app.route('/do_predict/',methods=['POST'])
def do_predict():
    jd = request.form['jd']
    X = A.get_single_X(jd)
    salary = clf.predict(X)
    return salary


if __name__ == '__main__':
    app.run(debug=True)
