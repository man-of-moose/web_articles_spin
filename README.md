# web_articles_spin
Analyzing web articles related to coronavirus to determine if they are Left, Center, or Right leaning

## Project Scope
The internet is a great place to get information and stay updated on the major happenings of the world. Unfortunately, in today's world it is up to the user to determine the degree of factfullness, or spin, of whatever content they've come across. Large news organizations and media companies compete for the largest viewerships, and often turn to dissemination of misinformation to push whatever agenda they may have. This is extremely problematic, especially when the misinformation has to do with unprecedented natural disasters affecting citizens lives.

This project's aim is to develop a NLP model to assist individuals in determining the spin of online web-articles, specifically pertaining to coronavirus news. The model will classify web articles as either Right-leaning, Center-leaning, or Left-leaning.

## Items of Note
- Analysis/: Has notebooks detailing data collection, EDA and Feature Engineering, as well as Model Building and Analysis
- pics/: All illustrations
- Scripts/: Streamlit script

## Procedure and Project Overview

### Collecting Data
- Selected 5+ media outlets for right, center, and left leaning targets
- Used NewsAPi to retrieve article URLs related to coronavirus
- Used Newspaper3k python library to scrape full article contents from URLs

![Image](Pics/class_imbalance.png?raw=true)

### Data Cleaning and EDA
- Pandas
- Numpy
- NLTK
- LDA

![Image](Pics/termite_plot.png?raw=true)

![Image](Pics/topic_by_target.png?raw=true)


### Feature Engineering
- Used texacity to generate article statistics
- Used textblob to generate article sentiment
- Used gensim to generate LDA topics (used as additional features)
- TFIDF, Word2Vec, Doc2Vec, FeatureUnion
- Used custom functions to determine profanity index

![Image](Pics/pol_by_target.png?raw=true)

![Image](Pics/coleman_index.png?raw=true)

## Results

- Addressed class imbalance by upsampling right to far_right, and by downsampling everything else to far_right
- Baseline Dummy Classifier accuracy of 24%
- Tested multiple models including SVM, RFC, MNB, Bagging, PAC, Word2Vec, LSTM, XBG
- Best Model: XGBoost (accuracy of 90%)

### Model Results
![Image](Pics/model_evaluation.png?raw=true)


### XGB Confusion Matrix
![Image](Pics/xgb_confusion.png?raw=true)


### XGB Feature Importance
![Image](Pics/xgb_features.png?raw=true)


