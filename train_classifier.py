import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Args:
    database_filepath (str): File path for the SQLite database file.

    Returns:
    X (DataFrame): Feature variables (message text).
    Y (DataFrame): Target variables (categories).
    category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    
    X = df['message']  # Messages (features)
    Y = df.iloc[:, 4:]  # Categories (target)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def build_model():
    """
    Build a machine learning pipeline for multi-output classification.

    Returns:
    model (Pipeline): A scikit-learn pipeline for processing and classifying messages.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),  # Text to word frequency counts
        ('tfidf', TfidfTransformer()),  # Term frequency-inverse document frequency
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classification
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on the test data and print classification report for each category.

    Args:
    model (Pipeline): Trained model pipeline.
    X_test (DataFrame): Test data (messages).
    Y_test (DataFrame): True values for test data (categories).
    category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model (Pipeline): Trained model.
    model_filepath (str): File path to save the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to load data, train a classifier, and save the model as a pickle file.

    This function:
    - Loads data from an SQLite database.
    - Trains a machine learning model on the data.
    - Saves the trained model as a pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data from {database_filepath}...')
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f'Saving model to {model_filepath}...')
        save_model(model, model_filepath)
        
        print('Model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
