import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # return the merged dataset
    return messages.merge(categories, how='inner', on=["id"])


def clean_data(df):
    
    # create a dataframe by splitting the 36 categories
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of categories dataframe and use as column names of categories
    row = categories.loc[0]
    categories.columns = row.apply(lambda x: x[:-2])
    
    # set each value to be the last character of the string and convert to numeric
    for column in categories:
    	categories[column] = categories[column].str[-1].astype(int)
    
    # replace categories column in df with dataframe categories
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove the duplicates
    df.drop_duplicates(subset=["id", "message"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # return the cleaned dataframe df
    return df


def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///{}'.format(database_filename))  
    df.to_sql('disaster', engine, index=False, if_exists='replace')


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
