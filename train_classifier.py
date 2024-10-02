import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import uniform
import pickle

def load_data(database_filepath):
    # Load data from the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)

    if 'message' not in df.columns:
        raise ValueError("Column 'message' not found in the Message table.")

    category_columns = df.columns[1:]
    X = df['message']  # Messages (features)
    Y = df[category_columns]  # Categories (target)

    # Convert all columns in Y to numeric type, handling errors
    for column in Y.columns:
        Y[column] = pd.to_numeric(Y[column], errors='coerce')

    # Fill NaN values with 0 to avoid mismatches in row counts
    Y = Y.fillna(0)

    # Sample data to reduce memory usage
    df = df.sample(frac=0.2, random_state=42)  # Use 20% of the dataset

    category_names = category_columns.tolist()

    return X, Y, category_names

def build_model():
    # Build a pipeline with TfidfVectorizer and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),  # Limit number of features to 10k
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=100)))
    ])

    # Hyperparameters for tuning the Logistic Regression
    param_grid = {
        'clf__estimator__C': uniform(0.1, 10),  # Uniform distribution for C parameter
        'clf__estimator__solver': ['liblinear', 'saga'],  # Different solvers
        'clf__estimator__penalty': ['l2', 'none']  # Different penalty terms
    }

    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_grid, 
        n_iter=5,  # Reduce iterations to minimize memory usage
        cv=3,  # 3-fold cross-validation
        verbose=2,
        n_jobs=-1  # Use all available processors
    )

    return random_search

def evaluate_model(model, X_test, Y_test, category_names):
    # Evaluate the model on each category
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    # Save the model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    # Manually specify the paths
    database_filepath = '/content/DisasterResponse.db'  # Path to your database file
    model_filepath = '/content/classifier.pkl'  # Path to save the model

    print(f'Loading data from {database_filepath}...')
    X, Y, category_names = load_data(database_filepath)

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model with RandomizedSearchCV...')
    model.fit(X_train, Y_train)

    print('Best parameters found:', model.best_params_)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print(f'Saving model to {model_filepath}...')
    save_model(model, model_filepath)

    print('Model saved!')

if __name__ == '__main__':
    main()
