## Data tools to process school data

# sys imports
import sys, os, shutil, errno

# string/data inputs
import string, csv, json, fileinput

# gucci imports
import numpy as np
import pandas as pd
import sklearn as skl

# misc
import progressbar
import collections
import pickle #for saving data objects


def main():
	print ("main")
	df = load_csv('../../data/massachusetts-public-schools-data/MA_Public_Schools_2017.csv')
		
def load_csv(filename):
	df = pd.read_csv(filename, sep =',')
	input_cols = list(df.columns)[:62]
	process(df)
	return df 

def process(frame):
	'''
	A. Split into elementary, middle, high schools

	'''
	categories = {
		'descriptive': ['School Code', 
						'School Name', 
						'Town', 
						'State', 
						'Zip', 
						'District Name',
						'District Code'],
		'endogeneous_input': [
			'School Type',
			'PK_Enrollment',
			'K_Enrollment',
			'1_Enrollment',
			'2_Enrollment',
			'3_Enrollment',
			'4_Enrollment',
			'5_Enrollment',
			'6_Enrollment',
			'7_Enrollment',
			'8_Enrollment',
			'9_Enrollment',
			'10_Enrollment',
			'11_Enrollment',
			'12_Enrollment',
			'SP_Enrollment',
			'TOTAL_Enrollment',
			'First Language Not English',
			'%% First Language Not English',
			'English Language Learner',
			'%% English Language Learner',
			'Students With Disabilities',
			'%% Students With Disabilities',
			'High Needs',
			'%% High Needs',
			'Economically Disadvantaged',
			'%% Economically Disadvantaged',
			'%% African American',
			'%% Asian',
			'%% Hispanic',
			'%% White',
			'%% Native American',
			'%% Native Hawaiian, Pacific Islander',
			'%% Multi-Race, Non-Hispanic',
			'%% Males',
			'%% Females',
			'Number of Students'
		],
		'exogeneous_input': [
			'Total # of Classes',
			'Average Class Size',
			'Salary Totals',
			'Average Salary',
			'FTE Count',
			'In-District Expenditures',
			'Total In-district FTEs',
			'Average In-District Expenditures per Pupil',
			'Total Expenditures',
			'Total Pupil FTEs',
			'Average Expenditures per Pupil'
		],
		'output_markers': []
	}

	feature_categories = categories['endogeneous_input'] + categories['exogeneous_input']
	print (',\n'.join(feature_categories))







if __name__ == '__main__':
	main()