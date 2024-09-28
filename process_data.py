# -*- coding: utf-8 -*-

# Necessary Imports
import pandas as pd
from sklearn.utils import resample
from sqlalchemy import create_engine

# Load datasets
categories = pd.read_csv('/content/categories.csv')
messages = pd.read_csv('/content/messages.csv')

# Drop duplicates and unnecessary columns
messages.drop(['original'], axis=1, inplace=True)
messages.drop_duplicates(inplace=True)

# Process categories dataset
cat = categories['categories'].str.split(';', expand=True)
row = cat.iloc[0]
category_colnames = row.apply(lambda x: x[:-2])
cat.columns = category_colnames

# Convert category values to integers (0 or 1)
for column in cat:
    cat[column] = cat[column].str[-1].astype(int)

# Drop the old 'categories' column and merge the processed categories
categories.drop('categories', axis=1, inplace=True)
categories = pd.concat([categories, cat], axis=1)

# Merge datasets on 'id'
df = messages.merge(categories, on='id')

# Drop nulls
df.dropna(inplace=True)

# Handle data imbalance
# Identify the columns to use for balancing
numerical_columns = df.select_dtypes(include='number').columns.tolist()

# Count the number of occurrences of each unique row to find majority/minority
counts = df[numerical_columns].value_counts()
majority_count = counts.max()
minority_count = counts.min()

# Separate majority and minority classes
majority_class = df[df[numerical_columns].apply(tuple, axis=1).isin(counts[counts == majority_count].index)]
minority_class = df[df[numerical_columns].apply(tuple, axis=1).isin(counts[counts == minority_count].index)]

# Changed from downsampling to upsampling as the majority class was too small
minority_class_upsampled = resample(minority_class,
                                     replace=True,  # sample with replacement
                                     n_samples=len(majority_class),  # to match majority class
                                     random_state=42)  # reproducible results

# Combine upsampled minority class with majority class
df_balanced = pd.concat([majority_class, minority_class_upsampled])

# SQL Database Creation
engine = create_engine('sqlite:///messages_categories.db')
df_balanced.to_sql('messages_categories', con=engine, if_exists='replace', index=False)

print("ETL process completed with nulls dropped and data imbalance handled. Data saved to SQL database.")

df.head()
