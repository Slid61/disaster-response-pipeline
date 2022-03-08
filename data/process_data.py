import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        messages_filepath: a string of the filepath of the csv containing messages
        categories_filepath: a string of the filepath of the csv containing message categories
        
    OUTPUT:
        df: a dataframe of the two input files merged.
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')
    return df


def clean_data(df):
    '''
    clean_data splits the categories column in the input DataFrame into individual columns,
    replacing the old column with the new ones in a ML-friendly format.
    INPUT:
        df: a pandas DataFrame object obtained from the load_data function.
        
    OUTPUT:
        clean_df: the same pandas DataFrame object, cleaned.
    '''
    # create the category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # obtain names by selecting the first row
    row = categories.iloc[0]
    category_colnames = []
    for cell in row:
        category_colnames.append(cell[:-2])
        
    # rename the column names
    categories.columns = category_colnames
    
    # convert column contents to 0 or 1 integers
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype('int')
        
    # replace categories column in df with the new columns
    df.drop('categories', axis=1, inplace=True)
    clean_df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    clean_df.drop_duplicates(inplace=True)

    # this column has values of 2, replace them with modal values (1)
    clean_df['related'].replace(2, 1, inplace=True)
    
    return clean_df


def save_data(df, database_filename):
    '''
    save_data saves an inputed DataFrame with the specified filename as a sqlite database.
    INPUT:
        df: pandas DataFrame object
        database_filename: string with the name of the database

    OUTPUT:
        a sqlite database in the same directory with the specified parameters.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))

    df.to_sql(database_filename, engine, index=False)

                           


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