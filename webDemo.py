#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
import requests
import urllib

import Analyser as A
clf = A.get_clf()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        return render_template('predict.html',jd=request.form['jd'])
    if request.method == 'GET':
        return render_template('predict.html')

@app.route('/do_predict/',methods=['POST'])
def do_predict():
    try:
        jd = request.form['jd']
        jd = urllib.unquote(jd)
        X = A.get_single_X(jd)
        if len(X.nonzero()[0]) < 6:
            return "信息量太少，无法预测——0"
        salary = clf.predict(X)
        print(salary[0])
    except Exception,e:
        print(e)
    return str(salary[0])

@app.route('/aboutme/',methods=['GET'])
def aboutme():
    return render_template('aboutme.html')

if __name__ == '__main__':
    app.run(debug=True,port=8000,host='127.0.0.1')
