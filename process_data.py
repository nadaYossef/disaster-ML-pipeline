import pandas as pd
from sqlalchemy import create_engine

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
    Save the cleaned dataset into an SQLite database.

    Args:
    df (DataFrame): Cleaned dataset.
    database_filename (str): File path for the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    """
    Main function to load, clean, and save the data.

    This function:
    - Loads messages and categories datasets.
    - Cleans the merged data.
    - Saves the cleaned data into an SQLite database.
    """
    messages_filepath = 'messages.csv'
    categories_filepath = 'categories.csv'
    database_filename = 'DisasterResponse.db'

    messages, categories = load_data(messages_filepath, categories_filepath)
    df = clean_data(messages, categories)
    save_data(df, database_filename)

if __name__ == '__main__':
    main()
