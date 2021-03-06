import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    load_data:load model data

    arguments:
    database_filepath:filepath of the data

    returns:
    X:features of model data
    Y:labels of model data
    category_names:category names of cateories
    """
    engine =  create_engine('sqlite:///./'+database_filepath)
    table_name = database_filepath.split("/")[-1].split(".")[0]
    df = pd.read_sql("SELECT * FROM "+table_name, engine)
    X = df[['message']]
    X = X.message.values
    Y = df.drop(['id', 'original', 'genre', 'message'],axis=1)
    category_names = Y.columns
    Y = Y.values
    return X, Y, category_names

def tokenize(text):
    """
    tokenize:text processing,including text.lower,word_tokenize,WordNetLemmatizer

    arguments:text file need to be processed

    return:text processed
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    build_model:build model GridSearchCV object
    arguments:none
    return:GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = {
        'clf__estimator__n_neighbors': [5, 6]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate_model:print the multi-output classification results
    arguments:
    fitted model,X_test test features,Y_test real labels in test,category_names:category names
    returns:none
    """
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    """
    save_model:save model
    arguments:model:
    modeleed to be saved model_filepath:file path model saved
    returns:none
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()