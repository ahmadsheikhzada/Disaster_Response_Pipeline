import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(["stopwords", "punkt", "wordnet"])

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from warnings import filterwarnings
filterwarnings("ignore")

def load_data(database_filepath):
    """
    
    Parameters
    ----------
    database_filepath : TYPE

    Returns
    -------
    X : pandas dataframe
        
    Y : pandas dataframe
        

    """
    # read sql database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster", engine)
    
    # construct features and target dataframes
    df.drop(columns=["original"], inplace=True)
    df.dropna(inplace=True)
    X = df["message"]
    Y = df.iloc[:, 3:]
    Y["related"] = Y["related"].apply(lambda x: 1.0 if (x==2.0) else x)
    Y.drop(columns=["child_alone"], inplace=True)
    
    return X, Y, list(Y.columns)

def tokenize(text):
    """
    Apply preliminary text preparation such as tokenizaiton, 
    normalization, lemmatizaiton and removing stopwords.

    Parameters
    ----------
    text : text
        

    Returns
    -------
    list of clean tokens

    """
    
    # remove punctuations
    text = re.sub("[^a-zA-Z0-9_]", " ", text)
    
    # tokenize the text
    tokens = word_tokenize(text)
    
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
    
    # remove stopwords
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words('english')]

    return clean_tokens
    
def build_model():
    """
    Builds model pipeline.

    Returns
    -------
    model pipeline

    """
    
    pipeline = Pipeline([
    	('vect', CountVectorizer(tokenizer=tokenize)),
    	('tfidf', TfidfTransformer()),
#    	('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0, n_jobs=1)))
        ('clf', MultiOutputClassifier(LinearSVC()))
        ])
    
    # parameters for GridSearchCV
    #parameters = {'clf__estimator__n_estimators':[5, 10, 20],
#                 'clf__estimator__max_depth':[5, 10]}

    parameters = {'clf__estimator__penalty' : ['l1', 'l2'],
   		   'clf__estimator__C': [0.01, 1, 100]} 
   		  
    cv_clf = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=1)
    
    return cv_clf
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    

    Parameters
    ----------
    model : sklearn gridsearch best estimator for f1_micro score
       
    X_test : numpy.array
       
    Y_test : numpy.array
        
    category_names : list
        output labels

    Returns
    -------
    None.

    """
    
    # predict model on test set
    Y_pred = model.predict(X_test)
    
    # output the result of model prediciton for each category
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[:, i], Y_pred[:, i]), "\n")
    
    # overall score of model
    print("overall f1_micro score: {}".format(model.score(X_test, Y_test)))
    

def save_model(model, model_filepath):
    """
    

    Parameters
    ----------
    model : sklearn gridsearch best estimator
        
    model_filepath : string
        saves model as pickle file in the suggested path

    Returns
    -------
    None.

    """
    
    # save model to pickle file
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2)
        
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
