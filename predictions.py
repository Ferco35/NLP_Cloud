from textblob import TextBlob
import pickle
import os
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

file_name="model_file.pickle.circ"
with (open(file_name, "rb")) as f:
    while True:
        try:
            model=pickle.load(f)
        except EOFError:
            break


import pickle
file_name="vectoriseur_file.pickle.circ"
with (open(file_name, "rb")) as f:
    while True:
        try:
            vectorizer=pickle.load(f)
        except EOFError:
            break

tokenizer = RegexpTokenizer(r'\w+')
def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed


import en_core_web_sm

nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])

lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()

    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,
                                                             'a'))  # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'v'))  # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'n'))  # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'r'))  # Lemmatise adverbs
        else:
            lemmatized_text_list.append(
                lemmatizer.lemmatize(word))  # If no tags has been found, perform a non specific lemmatisation

    return " ".join(lemmatized_text_list)

def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])

import contractions
def contraction_text(text):
    return contractions.fix(text)

negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"


def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i + 1 for i in range(len(tokens) - 1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx] = negative_prefix + tokens[idx]

    tokens = [token for i, token in enumerate(tokens) if i + 1 not in negative_idx]

    return " ".join(tokens)


from spacy.lang.en.stop_words import STOP_WORDS


def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]

    return " ".join([word for word in text.split() if word not in english_stopwords])


def preprocess_text(text):
    # Tokenize review
    text = tokenize_text(text)

    # Lemmatize review
    text = lemmatize_text(text)

    # Normalize review
    text = normalize_text(text)

    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)

    # Remove stopwords
    text = remove_stopwords(text)

    return text

labels = {
    #
    0 : "Mauvaise assistance téléphone" ,
    #
    1 : "Nourriture pas bonne" ,
    #
    2 : "Pizza" ,
    #
    3 : "Chicken" ,
    #
    4 : "Mauvaise qualité nourriture" ,
    #
    5 : "Service déplorable" ,
    #
    6 : "Burgers" ,
    #
    7 : "Trop d'attente" ,
    #
    8 : "Mauvaise expérience générale" ,
    #
    9 : "Nourriture pas bonne" ,
    #
    10 : "Mauvaise Livraison" ,
    #
    11 : "Mauvaises expériences répétitives" ,
    #
    12 : "Peronnel désorganisé" ,
    #
    13 : "Suchis" ,
    #
    14 : "Endroit dangereux"

}

import pandas as pd
import numpy as np

def predict(text,n_topics):
  n_topics=int(n_topics)
  blob = TextBlob(text)
  blob = blob.sentiment
  value = blob[0]
  if(value>0):
    return value, None
  else:
    print("Polarité : ", value," - COMMENTAIRE NEGATIF")
    text = preprocess_text(text)
    text=[text]
    X = vectorizer.transform(text)
    topics_correlations = model.transform(X)
    unsorted_topics_correlations=topics_correlations[0].copy()
    topics_correlations[0].sort()
    sorted=topics_correlations[0][::-1]
    topics=[]
    for i in range(n_topics):
      corr_value= sorted[i]
      result = np.where(unsorted_topics_correlations == corr_value)[0]
      topics.append(labels.get(result[0]))
    return value,topics

