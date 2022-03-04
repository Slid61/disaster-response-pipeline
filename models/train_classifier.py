import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import multilabel_confusion_matrix, classification_report, recall_score, make_scorer, accuracy_score, precision_score, f1_score 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine

import joblib

def load_data(database_filepath):
    ''' A function made to grab local data from a sqlite database and prepare it for ML processes.
    INPUT: database_filepath: a string of the path of the file to be pulled.
    OUTPUTS: X, y: pandas DataFrame objects.
                X: Input variables.
                y: output variables.
             category_names: The category names for the columns in y
    '''
    # read in file
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, con=engine)
    
    # clean data
    df.drop('child_alone', axis=1, inplace=True)

    # define features and label arrays
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns.values

    return X, y, category_names


def tokenize(text):
    ''' Converts text into tokens for ML processing
    INPUT: text: a string with a series of words, ideally in natural language.
    OUTPUT: clean_tokens: a list of tokens extracted from the original text.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    ''' A function that creates a ML pipeline and outlines parameters for a grid search.
    INPUTS: none
    OUTPUT: model_pipeline. A scikit-learn GridSearchCV object ready to be fitted.
    '''
    
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
        ])

    # define parameters for GridSearchCV
    params_grid = [{
                    'clf': [MultiOutputClassifier(RandomForestClassifier())],
                    'clf__estimator__min_samples_split': [5, 10],
                    'clf__estimator__max_depth': [5, 15, 25],
                    'clf__estimator__max_features': [5, 10],
                    'clf__estimator__n_estimators': [100, 200],
                    'clf__estimator__class_weight': ['balanced'],
                    }]
    scorers = {     'precision_score': make_scorer(precision_score, average='macro'),
                    'recall_score': make_scorer(recall_score, average='macro'),
                    'accuracy_score': make_scorer(accuracy_score),
                    'f1_score': make_scorer(f1_score, average='macro')
                    }
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=params_grid, scoring=scorers, refit='f1_score', 
                      error_score='raise', return_train_score=True, verbose=3, n_jobs=-1)

    return model_pipeline


def evaluate_model(model, X_test, y_test, category_names):
    ''' Prints out performance metrics for a ML model with a predict method.
    INPUTS: model: A scikit-learn Pipeline or GridSearchCV object that has already been fitted.
            X_test: pandas DataFrame containing the validation portion for out input features of the train-test-split.
            y_test: pandas DataFrame containing the validation portion for out output features of the train-test-split.
    OUTPUTS: None.
    '''
    y_pred = cv.predict(X_test)
    labels = np.unique(y_pred)
    confusion_mat = multilabel_confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Model Accuracy:", accuracy.mean())
    print("Accuracy Breakdown:", accuracy)
    print("\nBest Parameters:", cv.best_params_)

    for i, cat in enumerate(category_names):
        print(cat)
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))



def save_model(model, model_filepath):
    ''' A function that exports model results as a pickle file.
    INPUTS: model: A fully modeled and fitted GridSearchCV object.
            model_filepath: a string with the desired filepath to save the model.
    OUTPUT: A pickle file of the model named '{model_filepath}.pkl'
    '''
    # Export model as a pickle file
    joblib.dump(model, 'pipeline.pkl')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
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