import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''  
    loads the csv files from the given paths
    
    Args:
        messages_filepath : path of message.csv
        categories_filepath: path of categories.csv
    
    Returns:
        merged_df containing messages and categories data
    '''
    
    # loading data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merging them by id
    df_merged = messages.merge(categories, how='inner', on=['id'])
    
    return df_merged


def clean_data(df):
    ''' 
        Cleans the dataframe
         - renames columns of different categories
         - drops duplicates

         Args:
            df : Merged dataframe containing data from messages.csv and categories.csv

         Returns:
            df: Processed dataframe
    '''
    # split categories into separate category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # select the first row of the categories dataframe to help rename the columns 
    row = categories.iloc[0,:].tolist()
    
    # Renaming the column names using the first row
    category_colnames = [col_name[:-2] for col_name in row ]
    categories.columns = category_colnames
    
    #replacing original values with 1 and 0 (encoding values)
    for col in categories:
        # set each value to be the last character of the string
        categories[col] = categories[col].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[col] = pd.to_numeric(categories[col])

    #drop the categories column
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    #drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_name):
    '''
    Saves the dataframe into a database
    
    Args:
        df: dataframe to be saved
        database_name : name of the database file
        
    Returns:
        None
    '''

    engine = create_engine('sqlite:///'+database_name)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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