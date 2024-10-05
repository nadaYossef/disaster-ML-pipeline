import pandas as pd
from sqlalchemy import create_engine
import pickle
import os


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets from CSV files.

    Args:
    messages_filepath (str): File path for the messages CSV file.
    categories_filepath (str): File path for the categories CSV file.

    Returns:
    messages (DataFrame): Loaded messages dataset.
    categories (DataFrame): Loaded categories dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories

def clean_data(messages, categories):
    """
    Merge and clean the messages and categories datasets.

    - Merges messages and categories on the 'id' column.
    - Splits the categories into separate columns.
    - Converts category values to binary (0 or 1).
    - Removes duplicates from the dataset.

    Args:
    messages (DataFrame): Messages dataset.
    categories (DataFrame): Categories dataset.

    Returns:
    df (DataFrame): Cleaned and merged dataset.
    """
    # Merge messages and categories datasets
    df = messages.merge(categories, on='id')

    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to binary (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Drop original categories column and concatenate cleaned categories
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataset into an SQLite database and as a pickle file.

    Args:
    df (DataFrame): Cleaned dataset.
    database_filename (str): File path for the SQLite database file.
    """
    # Save the dataset to an SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

    # Save the dataset as a pickle file
    with open('cleaned_data.pkl', 'wb') as file:
        pickle.dump(df, file)

    print("Data saved to SQLite database and as 'cleaned_data.pkl'")

def main():
    messages_filepath = os.path.join(os.path.dirname(__file__), 'data', 'messages.csv')
    categories_filepath = os.path.join(os.path.dirname(__file__), 'data', 'categories.csv')
    database_filename = os.path.join(os.path.dirname(__file__), 'DisasterResponse.db')

    # Load datasets
    messages, categories = load_data(messages_filepath, categories_filepath)
    
    # Clean datasets
    df = clean_data(messages, categories)
    
    # Save cleaned dataset
    save_data(df, database_filename)

if __name__ == '__main__':
    main()
