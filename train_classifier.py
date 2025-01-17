import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from tqdm import tqdm
from joblib import parallel_backend
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Enable tqdm for all iterations in scikit-learn
tqdm.pandas()

def tokenize(text):
    """
    Tokenize the text data by normalizing case, removing stopwords, and lemmatizing.

    Args:
    text (str): The text to be tokenized.

    Returns:
    tokens (list): List of cleaned tokens.
    """
    # Normalize case and tokenize text
    text = text.lower()
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return tokens

def load_data(database_filepath):
    """Load data from the SQLite database and display database information."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)

    # Skip the first column which is the message ID
    X = df['message']  # Messages (features)
    Y = df.iloc[:, 6:]  # Assuming the target columns start from the 6th column

    # Convert all columns in Y to integers, handling errors and filling NaNs with 0
    Y = Y.apply(lambda col: pd.to_numeric(col, errors='coerce').fillna(0).astype(int))
    
    # Ensure target variables are binary
    Y = (Y > 0).astype(int)

    # Print database information and show head
    print("Database loaded successfully!")
    print("Dataframe Info:\n")
    print(df.info())  # Show dataframe structure info
    print("\nDataframe Head:\n")
    print(df.head())  # Show the first 5 rows of the dataframe

    # Check unique values to confirm binary encoding
    print("\nUnique values in target columns:\n", Y.nunique())

    return X, Y, Y.columns.tolist()

def sample_data(X, Y, sample_size=0.1):
    """Sample a subset of the data."""
    X_sample, _, Y_sample, _ = train_test_split(X, Y, train_size=sample_size, random_state=42)
    return X_sample, Y_sample

def build_model():
    """Build a machine learning pipeline with RandomizedSearchCV."""
    pipeline = Pipeline([
        ('countvectorizer', CountVectorizer(tokenizer=tokenize, max_features=5000)),  # Use custom tokenizer
        ('clf', MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss')))
    ])

    # Hyperparameters for tuning the XGBoost
    param_dist = {
        'clf__estimator__max_depth': [3, 5],
        'clf__estimator__learning_rate': [0.01, 0.1],
        'clf__estimator__n_estimators': [100, 150]
    }

    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,  # Number of random samples
        cv=2,  # 2-fold cross-validation
        verbose=2,
        n_jobs=-1,  # Use all available processors
        random_state=42
    )

    return random_search

def fit_with_progress_bar(random_search, X_train, Y_train):
    """Fit the RandomizedSearchCV model and show a progress bar."""
    n_candidates = random_search.n_iter  # Total number of candidates
    pbar = tqdm(total=n_candidates, desc="Fitting models", unit="fit")

    # Define the callback to update progress bar
    random_search.fit(X_train, Y_train)

    pbar.close()

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model on the test set."""
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """Save the model as a pickle file."""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main(database_filepath, model_filepath):
    print(f'Loading data from {database_filepath}...')
    X, Y, category_names = load_data(database_filepath)

    # Sample data
    print('Sampling data...')
    X_sample, Y_sample = sample_data(X, Y, sample_size=0.1)  # Sample 10% of the data

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_sample, Y_sample, test_size=0.2, random_state=42)

    print('Building model...')
    model = build_model()

    print('Training model with RandomizedSearchCV...')
    fit_with_progress_bar(model, X_train, Y_train)

    print('Best parameters found:', model.best_params_)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print(f'Saving model to {model_filepath}...')
    save_model(model, model_filepath)

    print('Model saved!')

if __name__ == '__main__':
    # Manually specify the paths or use command-line arguments
    database_filepath = 'DisasterResponse.db'  # Path to your database file
    model_filepath = 'classifier.pkl'  # Path to save the model

    main(database_filepath, model_filepath)
