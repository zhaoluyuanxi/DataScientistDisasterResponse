import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load_data:
    load messages and categories data

    arguments:
    messages_filepath:the filepaths of messages data
    categories_filepath:the filepaths of categories data

    retruns:
    df:merged data df of messages and categories on ID
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])  
    return df
    


def clean_data(df):
    """
    clean_data:
    clean data by split categories to columns,converts categories values to binary values,drop dulicates values

    arguments:
    datdframe of unclened values 

    returns:
    dataframes of cleaned values
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = []
    for i in range(row.shape[0]):
        category_colnames.append(row[i].split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] =  pd.to_numeric(categories[column])
        
    df = pd.concat([df, categories], axis=1)
    df['related'][df['related']==2] = 1
              
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    save_data:save data to sqlite db

    arguments:
    df:dataframe need to be saved
    database_filename:data save filepaths

    return:
    None
    """
    engine = create_engine('sqlite:///./'+database_filename)
    table_name = database_filename.split("/")[-1].split(".")[0]
    df.to_sql(table_name, engine, index=False) 


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