#encoding: utf8

from flask import Flask
from flask import render_template
from flask.ext.bootstrap import Bootstrap


from flask.ext.wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required

class JdForm(Form):
    name = StringField('输入你的jd', validators=[Required()])
    submit = SubmitField('看看价格')

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def hello_world():
    return render_template('templates/p.html',form=JdForm())


if __name__ == '__main__':
    app.run(port=8000)
