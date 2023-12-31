import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from "messages" dataset and "categories" dataset.
    Input:
    + messages_filepath
    + categories_filepath
    Output: merged dataframe containing messages and categories data.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id", how="inner")
    return df
    
def clean_data(df):
    '''
    Clean data in the dataframe df
    '''
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0, :]
    categories.columns = row.apply(lambda x: x[:-2])
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns="categories")
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicated rows
    df = df.drop_duplicates()
    # drop rows with target values of non-binary type
    df = df[df["related"]<=1]
    return df

def save_data(df, database_filepath):
    '''
    Save dataframe to a table in the database_path.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql("messages_and_categories", engine, index=False, if_exists="replace")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
