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
    Load data from the SQLite database, extract features (messages) and target variables (categories).
    
    Args:
        database_filepath (str): File path of the SQLite database.
    
    Returns:
        X (pd.Series): Feature data (messages) from the 'message' column.
        Y (pd.DataFrame): Target variables (categories) for multi-output classification.
        category_names (list): List of category names representing the columns of Y.
    
    Raises:
        ValueError: If 'message' column is not found in the 'Message' table of the database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)

    if 'message' not in df.columns:
        raise ValueError("Column 'message' not found in the Message table.")

    category_columns = df.columns[1:]  # Assuming the first column is 'message' and others are categories
    X = df['message']  # Messages (features)
    Y = df[category_columns]  # Categories (target)
    
    # Ensure target columns are numeric, handle any conversion errors
    for column in Y.columns:
        Y[column] = pd.to_numeric(Y[column], errors='coerce')
    
    # Fill missing values in Y with 0
    Y = Y.fillna(0)
    
    category_names = category_columns.tolist()

    return X, Y, category_names

def build_model():
    """
    Build a machine learning pipeline for multi-output classification using RandomForestClassifier.
    
    Returns:
        pipeline (Pipeline): A Scikit-learn pipeline object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),  # Convert text to word frequency counts
        ('tfidf', TfidfTransformer()),  # Term frequency-inverse document frequency
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classification
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained model on test data and print classification reports for each category.
    
    Args:
        model (Pipeline): The trained model pipeline.
        X_test (pd.Series): Test features (messages).
        Y_test (pd.DataFrame): True labels for the test set (target categories).
        category_names (list): List of category names for the target variables.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
        model (Pipeline): Trained machine learning model.
        model_filepath (str): File path where the model pickle file will be saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to load data, build the model, train it, evaluate it, and save the model.
    
    - Loads the disaster response data from the SQLite database.
    - Splits the data into training and test sets.
    - Trains a machine learning model on the data.
    - Evaluates the model on the test set and prints the performance for each category.
    - Saves the trained model as a pickle file.
    """
    # File paths for the database and where to save the trained model
    database_filepath = '/content/DisasterResponse.db'  # Path to your SQLite database
    model_filepath = '/content/classifier.pkl'  # Path to save the model pickle

    print(f'Loading data from {database_filepath}...')
    X, Y, category_names = load_data(database_filepath)
    
    # Split data into training and test sets
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

if __name__ == '__main__':
    main()
