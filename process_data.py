import pandas as pd
from sqlalchemy import create_engine
import pickle
import os
import sys

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
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
        messages, categories = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(messages, categories)

        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")
    else:
        print("Please provide the filepaths of the messages and categories datasets as well as the filepath of the database to save the cleaned data to as arguments. \n\nExample: python process_data.py data/messages.csv data/categories.csv DisasterResponse.db")

if __name__ == '__main__':
    main()
