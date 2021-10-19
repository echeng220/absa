from operator import pos
import pandas as pd
import numpy as np
import nltk
import spacy
import os

from flask import Flask, render_template, request

from absa_functions import *

nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')

pd.set_option('display.width', 1000)
pd.set_option('colheader_justify', 'center')

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

@app.route('/')
def index():
	return render_template("absa.html")

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        rawtext = request.form['rawtext']

        df = pos_prediction(rawtext)

        df_chunk = pos_chunk_prediction(rawtext)

    return render_template('absa.html', title='Restaurant Reviews', tables=[df.to_html(classes='mystyle', header="true")],
                            tables2 = [df_chunk.to_html(classes='mystyle', header="true")])
  
if __name__ == "__main__":
    app.run(debug=True)