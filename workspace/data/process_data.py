import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

	''' Takes in messages filepath ,and categories filepath then merges them in pandas datframe and returns it
			Parameters:
                    messages_filepath (string): path where messages csv file is stored
                    categories_filepath (string): path where categories csv file is stored

            Returns:
                    df (pandas DataFrame): Dataframe made by merging categories and messages Datframes
    '''
    
	categories = pd.read_csv(categories_filepath)
	messages = pd.read_csv(messages_filepath)
	df = pd.merge(messages, categories, left_on='id', right_on='id', how='left')

	return  df


def clean_data(df):

	''' Cleans a datframe to make it suitable for further use
		Parameters:
				df(Pandas DataFrame) : Uncleaned Dataframe
		Return : 
				df(Pandas DataFrame) : Clean Dataframe
	
	'''

	categories = df['categories'].str.split(";", expand=True) 
	cols = list(categories.iloc[0].str.split("-", expand=True)[0])

	for col, name in zip(categories.columns, cols):
	    categories[col] = categories[col].str.split("-", expand=True)[1]
	categories.columns = cols

	df = pd.concat([df, categories], axis = 1)
	df = df.drop(['categories'], axis=1)

	df = df.T.drop_duplicates().T

	return df


def save_data(df, database_filename):

	''' Save Dataframe as SQL Table '''

	engine = create_engine('sqlite:///'+database_filename)
	df.to_sql('udacity_etl', engine, index=False)  


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