#encoding: utf8

from flask import Flask
from flask.ext.bootstrap import Bootstrap

bootstrap = Bootstrap(app)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(8080)
