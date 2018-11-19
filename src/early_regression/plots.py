## Data tools to process school data

# sys imports
import sys, os, shutil, errno

# string/data inputs
import string, csv, json, fileinput

# gucci imports
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt

# misc
import progressbar
import collections
import pickle #for saving data objects


def main():
	print ("main")
	school_data = load_csv('../../data/massachusetts-public-schools-data/MA_Public_Schools_2017.csv')
	scraped_data = load_csv('../../scraper/econ_full_scrape_11-17-2018.csv')
	school_process(school_data, scraped_data)
	## want a dictionary of index to school code

def load_csv(filename):
	df = pd.read_csv(filename, sep =',')
	return df 

def school_process(school_data, zip_data):
	'''
	A. Split into elementary, middle, high schools

	'''
	school_categories = {
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
			'% First Language Not English',
			'% English Language Learner',
			'% Students With Disabilities',
			'% High Needs',
			'% Economically Disadvantaged',
			'% African American',
			'% Asian',
			'% Hispanic',
			'% White',
			'% Native American',
			'% Native Hawaiian, Pacific Islander',
			'% Multi-Race, Non-Hispanic',
			'% Males',
			'% Females',
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

	zip_categories = [
		'Median household income',
		'Avg Hours Worked',
		'Public Assistance and SSI',
		'Unemployment Rate',
		'Labor Force Participation',
		'Percent of Population In Poverty',
		'Public Assistance Percent',
		'Gini Index',
		'Single Earner Families',
		'Families with No One Working',
		'Avg Commute Time',
		'Self Employment Income',
		'Total Self-employed Men',
		'Total Self-employed Women',
		'Zip Code',
		'Less Than Highschool in Poverty',
		'Local government',
		'State government'
	]

	school_cols = school_categories['descriptive'] + school_categories['endogeneous_input'] + school_categories['exogeneous_input']
	school_data = school_data[school_cols]
	one_hot_type = pd.get_dummies(school_data['School Type'])
	school_data = school_data.join(one_hot_type)
	school_data.drop('School Type', axis=1, inplace=True)
	school_data = school_data.rename(columns={'Zip':'Zip Code'})
	zip_data = zip_data[zip_categories]

	joined_input = school_data.join(zip_data, on='Zip Code', how='left', rsuffix='_scrape')

	input = 'Average Expenditures per Pupil' ####### @zane fill this in with your metric(s)-- indices should align with joined_input

	highschools = joined_input['12_Enrollment'] > 0

	x_cols = school_categories['endogeneous_input'] + school_categories['exogeneous_input'] + zip_categories
	full_y = high_schools[x_cols]
	data_dict = {
		'full_x': input,
		'full_y': full_y
	}
	

	return data_dict
	
	







if __name__ == '__main__':
	main()