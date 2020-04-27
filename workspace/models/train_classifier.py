import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report

def load_data(database_filepath):

    ''' Load SQL Table as Pandas Dataframe and Returs X and Y dataframe as data and labels respectively '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('udacity_etl', engine)
    X = df[df.columns[:4]]
    Y = df[df.columns[4:]]

    return X, Y

def tokenize(text):

    ''' Takes in text and returns list of tokens '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    ''' Returns pipeline which specifies order of data processing and then training '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test):

	''' Takes in model, Test Data ,and Test labels then returns accuracy of model on test data and prints classification report for each
	class in multi class output'''

	y_pred = model.predict(X_test)
	overall_accuracy = (y_pred == Y_test).mean().mean()
	print("Overall accuracy of our model is "+ str(overall_accuracy))
	
	category_names = Y_test.columns
	Y_test = Y_test.values.astype('int')
	y_pred = y_pred.astype('int')
	y_pred[y_pred > 1] = 1
	Y_test[Y_test > 1] = 1
	print("The f1 score, precision and recall for the test set is outputted for each category")
	print(classification_report(Y_test, y_pred, target_names=category_names))



def save_model(model, model_filepath):

    # Save to file in the current working directory
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')

        model.fit(X_train['message'], Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test['message'], Y_test)

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