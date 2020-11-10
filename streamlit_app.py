#-*- coding: utf-8 -*-


#import libraries
import pickle
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

from newspaper import Article, Config
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import textacy
import re
import string
import scipy
from textblob import TextBlob
seed=42
import streamlit as st

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
lemmatizer = WordNetLemmatizer()



#set stopwords and punctuations
stopwords = stopwords.words('english')
stopwords += list(string.punctuation)
stopwords += ["n't","''","'re'","”","``","“","''","’","'s","'re","http","https","char","reuters","wall","street","journal","photo"]

#set config for Newspaper3k
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent

#get text
def get_full_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        full_text = article.title + " " + article.text
        return full_text
    except:
        return '401 Error'


#preprocessing
def black_txt(token):
    return token not in stopwords and token not in list(string.punctuation) and len(token) > 3

def clean_txt(text, string=True):
    clean_text = []
    clean_text2 = []
    text = text.lower()
    text = re.sub("'", "", text)
    text = re.sub("\n", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = re.sub('time magazine', '', text)
    text = re.sub('breitbart', '', text)
    text = re.sub('click', '', text)
    clean_text = [lemmatizer.lemmatize(word, pos="v") for word in word_tokenize(text) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    if string == True:
        return " ".join(clean_text2)
    else:
        return clean_text2


#feature engineering
def get_n_long_words(text):
    doc = textacy.make_spacy_doc(text, lang=nlp)
    ts = textacy.TextStats(doc)
    n_words = ts.n_words
    return ts.n_long_words / n_words

def get_n_monosyllable_words(text):
    doc = textacy.make_spacy_doc(text, lang=nlp)
    ts = textacy.TextStats(doc)
    n_words = ts.n_words
    return ts.n_monosyllable_words / n_words

def get_n_polysyllable_words(text):
    doc = textacy.make_spacy_doc(text, lang=nlp)
    ts = textacy.TextStats(doc)
    n_words = ts.n_words
    return ts.n_polysyllable_words / n_words

def get_n_unique_words(text):
    doc = textacy.make_spacy_doc(text, lang=nlp)
    ts = textacy.TextStats(doc)
    n_words = ts.n_words
    return ts.n_unique_words / n_words

def polarity_txt(text):
    return TextBlob(text).sentiment[0]

def subj_txt(text):
    return  TextBlob(text).sentiment[1]

def mean_sentences_per_100(text, group_size):
    sentences = []
    text_split = text.split(" ")
    length = len(text_split)
    i = 0
    while i < length:
        values = []
        for word in text_split[i:i+group_size]:
            values.append(word)
        sentence_counter = 0
        for word in values:
            if '.' in word:
                sentence_counter += 1
        sentences.append(sentence_counter)
        i = i + group_size
    return sum(sentences)/len(sentences)

def mean_character_per_100(text, group_size):
    averages = []
    text_split = text.split(" ")
    length = len(text_split)
    i = 0
    while i < length:
        values = []
        for word in text_split[i:i + group_size]:
            values.append(len(word))
        if len(values) == group_size:
            averages.append(sum(values) / len(values))
        else:
            pass
        i = i + group_size
    return sum(averages) / len(averages)

def cli(text, group_size):
    mccphw = mean_character_per_100(text, group_size)
    mscphw = mean_sentences_per_100(text, group_size)
    return (0.0588 * mccphw) - (0.296 * mscphw) - 15.8



class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, data):
        return [{'pol': row['polarity'],
                 'sub': row['subjectivity'],
                 'n_long_words': row['n_long_words'],
                 'n_monosyllable_words': row['n_monosyllable_words'],
                 'n_polysyllable_words': row['n_polysyllable_words'],
                 'n_unique_words': row['n_unique_words'],
                 'coleman_index': row['coleman_index']} for _, row in data.iterrows()]

    
#load pickles
normalizer = pd.read_pickle('normalizer.pickle')
pipe = pd.read_pickle('pipeline.pickle')
model = pd.read_pickle('model.pickle')

#execute code
# def main():
#     try:
#         st.title("Covid19 Political Bias Detector")
#         st.text('Pass a web article (URL) related to COVID19 below')
#         web_article = st.text_input("Paste URL Here!")
#         data = {'predictor': get_full_text(web_article)}
#         data['n_long_words'] = get_n_long_words(data['predictor'])
#         data['n_monosyllable_words'] = get_n_monosyllable_words(data['predictor'])
#         data['n_polysyllable_words'] = get_n_polysyllable_words(data['predictor'])
#         data['n_unique_words'] = get_n_unique_words(data['predictor'])
#         data['coleman_index'] = cli(data['predictor'],100)
#         data['polarity'] = polarity_txt(data['predictor'])
#         data['subjectivity'] = subj_txt(data['predictor'])
#         data = pd.DataFrame(data, index=[0])
#         data[['polarity', 'coleman_index']] = normalizer.fit_transform(data[['polarity', 'coleman_index']])
#         train_vec = pipe.transform(data)
#         pred = model.predict(train_vec)[0]
#         st.write("Based on the contents, I believe this article leans: {}".format(pred))
#     except:
#         print("Error. Either article is not long enough, or website is down")

try:
    st.title("Covid19 Political Bias Detector")
    st.text('Pass a web article (URL) related to COVID19 below')
    data = {'predictor': get_full_text(st.text_input("Paste URL Here!"))}
    st.write('retrieving n_long_words')
    data['n_long_words'] = get_n_long_words(data['predictor'])
    st.write('retrieving n_monosyllable_words')
    data['n_monosyllable_words'] = get_n_monosyllable_words(data['predictor'])
    st.write('retrieving n_polysyllable_words')
    data['n_polysyllable_words'] = get_n_polysyllable_words(data['predictor'])
    st.write('retrieving n_unique_words')
    data['n_unique_words'] = get_n_unique_words(data['predictor'])
    st.write('retrieving coleman')
    data['coleman_index'] = cli(data['predictor'],100)
    st.write('retrieving polarity')
    data['polarity'] = polarity_txt(data['predictor'])
    st.write('retrieving subjectivity')
    data['subjectivity'] = subj_txt(data['predictor'])
    data = pd.DataFrame(data, index=[0])
    st.write(data)
    data[['polarity', 'coleman_index']] = normalizer.fit_transform(data[['polarity', 'coleman_index']])
    st.write(data)
    st.write("Initiating Pipeline Transformation")
    train_vec = pipe.transform(data)
    st.write("Initializing Model Prediction")
    pred = model.predict(train_vec)[0]
    st.write('prediction created')
    st.write("Based on the contents, I believe this article leans: {}".format(pred))
except:
    st.write("Error. Either article is not long enough, or website is down")
    
# if __name__ == "__main__":
#     main()

# try:
#     st.title("Election emoji guessing machine!")
#     st.text('Type an election related tweet in the box below:')
#     data = {'tweet': st.text_input("Enter a tweet! ")}
#     data['sentiment_score'] = sentiment_scores(data['tweet'])
#     data['sentiment_score'] = normalizer.transform(np.array(data['sentiment_score']).reshape(1, -1))[0][0]
#     data['exclamation_points'] = exclamation_percentage(data['tweet'])
#     data['capitalization'] = capital_percentage(data['tweet'])
#     data['profanity'] = check_profanity(data['tweet'])
#     data['subjectivity'] = get_subjectivity(data['tweet'])
#     data = pd.DataFrame(data, index=[0])
#     train_vec = pipe.transform(data)
#     pred = model.predict(train_vec)[0]
#     st.write(f"I am guessing your tweet can represented with this: {pred}")
# except:
#     pass








