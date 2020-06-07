# import libraries
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump, load
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def load_data(database_filepath):
    '''
    loads data from the database
    
    Args:
        database_filepath : filepath to the databse
        
    Returns:
        X: 'message' column 
        y: one-hot encoded categories
        category_names: category names in y
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages;', con = engine)

    # splitting the target
    X = df['message']

    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    categories_names = y.columns
    
    return X, y, categories_names


def tokenize(text):
    ''' 
    Tokenizes the text into words, nomalizes it and performs lemmatization
    
    Args: 
        text: Raw Text
        
    Returns:
        clean_tokens: Lemmatized tokens containing only alphanumeric characters 
    '''
    
    # removing non-alphanumeric characters
    text = re.sub('\W', '', text)
    
    # tokenizing text
    tokens = word_tokenize(text)
    
    # lematization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token, 'n').lower().strip() for token in tokens]
    clean_tokens = [lemmatizer.lemmatize(token, 'v').lower().strip() for token in tokens]
    
    return clean_tokens


def build_model():
    '''
    Builds the ML pipeline
    
    Args:
        None
    Return:
        cv - Result of the GridSearchCV model
    '''

    
    params = {'clf__estimator__estimator__C': [1,2,4,8]}

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
        ])
    pipeline.get_params().keys()
    cv = GridSearchCV(pipeline, param_grid=params, n_jobs=-1, cv=5)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    ''' 
    Evaluates the model, prints accuracy and the classification reports
    
    Args:
        model: pipeline object
        X_test: Test features
        Y_test: Test labels
        category_names: names of categories present in the dataset
    
    Returns:
        None 
    '''
    
    y_pred = model.predict(X_test)
    print("Accuracy:", (y_pred == y_test).sum()/y_test.shape[0])
    
    # Classification Report
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    for i in range(len(y_pred.columns)):
        print(y_pred.columns.tolist()[i])
        print(classification_report(y_test.iloc[:,i].tolist(), y_pred.iloc[:,i].tolist()))


def save_model(model, model_filepath):
    ''' 
    Saves the model for later use
    
    Args:
        model: Pipeline object
        model_filepath: model name
        
    Returns:
        None
    '''
    
    dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=1)
        
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
