from operator import pos
import pandas as pd
import numpy as np
import nltk
import spacy
import pickle
import os

from flask import Flask, render_template, url_for, flash
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')

from forms import ReviewForm
from absa_functions import *

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

posts = [
    {
        'author': 'evan',
        'title': 'blog post 1',
        'content': '1st post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'cheng',
        'title': 'blog post 2',
        'content': '2nd post content',
        'date_posted': 'April 7, 2020'
    }
]

@app.route('/', methods=['GET', 'POST'])
def run_model():
    form = ReviewForm('/')
    
    # pred = pos_prediction()

    return render_template('absa.html', title='Restaurant Reviews', form=form)
  
if __name__ == "__main__":
    app.run(debug=True)