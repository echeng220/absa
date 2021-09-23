import xml.etree.ElementTree as et
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser  = English()

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

import gensim
from gensim import corpora
from gensim.models import TfidfModel
from gensim import matutils
from gensim.models import CoherenceModel

from operator import itemgetter

def encode_category(sentiments, category):
    """Return the sentiment for a specific category in a dictionary where the keys are the categories, 
    and the values are the sentiments for that category."""
    if category in sentiments.keys():
        output = sentiments[category]
    else:
        output = None
    return output

def create_corpus(xml_path):
    tree = et.parse(xml_path)
    root = tree.getroot()

    m = len(root) - 1

    #For every training example, we collect sentiments for every category, adding the text and sentiments to the "corpus".
    #We also find all the unique categories in the training set, appending the to the "categories" list

    corpus = {}
    categories = []

    for i in range(m):
        doc = {}
        doc.update({'text': root[i][0].text})
        tmp = {}
        for j in range(len(root[i][1])):
            if root[i][1][j].attrib['category'] not in categories:
                categories.append(root[i][1][j].attrib['category'])
            tmp.update({root[i][1][j].attrib['category']: root[i][1][j].attrib['polarity']})
        doc.update({'sentiment': tmp})
        corpus.update({i: doc})

    corpus_df = pd.DataFrame(corpus).transpose()

    for cat in categories:
        corpus_df[f'{cat}'] = corpus_df['sentiment'].apply(lambda x: encode_category(x, cat))

    return corpus_df, categories

def tokenize(text):
    """Creates tokens for LDA model. Passes over all whitespaces, adds special tokens for URLs and screen names. 
    Puts all tokens in lower case."""
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens 

def get_lemma(word):
    """Lemmatize (get root word) for a given word."""
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    """More lemmatization."""
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    """Generate list of tokens, keeping all tokens that are >2 characters, and are not in the stopword list.
    Lemmatize each token."""
    en_stop = set(nltk.corpus.stopwords.words('english'))

    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2 and "'" not in token]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def preprocess_corpus(corpus_df):
    text_data = []
    for i in range(len(corpus_df)):
        tokens = prepare_text_for_lda(corpus_df['text'][i])
        text_data.append(tokens)
    return text_data

def create_bagofwords(text_data):
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    return dictionary, corpus

def apply_lda_to_corpus(lda_model, corpus, corpus_df, categories):
    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
    topic_strengths = [[tuple[1] for tuple in doc_topics[i]] for i in range(len(doc_topics))]

    topic_strengths_df = pd.DataFrame(topic_strengths)
    topic_strengths_df.columns = [f'latent_{i}' for i in range(0, len(categories) - 1)]

    topic_match_df = corpus_df[categories].merge(topic_strengths_df, left_index=True, right_index=True)
    topic_match_df = topic_match_df.replace(['positive', 'neutral', 'negative'], 1)
    topic_match_df = topic_match_df.fillna(0)
    return topic_match_df

def get_strong_topics(document, topic_map):
    """Choose the strongest topics for each document, such that the number of topics chosen matches
    the number of training labels assigned to the document."""
    
    num_topics = document[0]
    topic_dict = {topic_map[topic[0]]: topic[1] for topic in document[-1]}
        
    output = dict(sorted(topic_dict.items(), key = itemgetter(1), reverse = True)[0:num_topics])
    
    return output      

def compile_strong_topics(lda_model, corpus, corpus_df, topic_map):
    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
    label_counts = list(corpus_df['sentiment'].apply(lambda x: len(x)))
    doc_topics = tuple(zip(label_counts, doc_topics))

    strong_topics = []
    
    for doc in doc_topics:
        tmp = get_strong_topics(document = doc, topic_map = topic_map)
        strong_topics.append(tmp)  

    lda_df = pd.DataFrame({'text': corpus_df['text'], 'lda_topic': strong_topics})
    lda_df
    return lda_df

def sentiment_analysis(lda_df, categories):
    sid = SentimentIntensityAnalyzer()
    lda_df['vader_score'] = lda_df['text'].apply(lambda review: sid.polarity_scores(review))
    lda_df['compound_score'] = lda_df['vader_score'].apply(lambda score_dict: score_dict['compound'])
    lda_df['sentiment'] = lda_df['compound_score'].apply(lambda sent: 'positive' if sent > 0 else ('neutral' if sent ==0 else 'negative'))

    for cat in categories:
        lda_df[f'{cat}'] = lda_df['lda_topic'].apply(lambda x: encode_category(x, cat))
        lda_df[f'{cat}'] = lda_df[f'{cat}'].mask(lda_df[f'{cat}'].notnull(), lda_df['sentiment'])

    return lda_df

def create_label_df(df, categories):
    label_df = df[categories].copy()
    label_df.replace(to_replace = [None], value = 'N/A', inplace = True)
    return label_df

def calculate_accuracy(pred_cats_df, true_cats_df):
    diff_df = pred_cats_df.eq(true_cats_df).astype(int)
    return diff_df.mean().mean() * 100

def add_label_col(df, categories):
    enum_df = MultiColumnLabelEncoder(columns=categories).fit_transform(df)
    enum_df['labels'] = enum_df[categories].values.tolist()
    return enum_df

def calculate_scores(enum_preds_df, enum_true_df, categories):
    y_true = np.stack(enum_true_df.labels.values)
    y_pred = np.stack(enum_preds_df.labels.values)

    m = MultiLabelBinarizer().fit(y_true)

    precision = precision_score(m.transform(y_true), m.transform(y_pred), average='weighted')
    recall = recall_score(m.transform(y_true), m.transform(y_pred), average='weighted')
    f1 = f1_score(m.transform(y_true), m.transform(y_pred), average='weighted')

    # report = classification_report(m.transform(y_true), m.transform(y_pred), labels=categories, target_names=categories)
    # print(report)

    return precision, recall, f1


class MultiColumnLabelEncoder:
    def __init__(self,columns=None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)



