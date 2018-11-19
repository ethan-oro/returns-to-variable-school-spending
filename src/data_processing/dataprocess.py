## Data tools to process school data

# sys imports
import sys, os, shutil, errno

# string/data inputs
import string, csv, json, fileinput

# gucci imports
import numpy as np
import pandas as pd

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

def grab_data():
	school_data = load_csv('../../data/massachusetts-public-schools-data/MA_Public_Schools_2017.csv')
	scraped_data = load_csv('../../scraper/econ_full_scrape_11-17-2018.csv')
	return school_process(school_data, scraped_data)

def grab_data_spend():
	school_data = load_csv('../../data/massachusetts-public-schools-data/MA_Public_Schools_2017.csv')
	scraped_data = load_csv('../../scraper/econ_full_scrape_11-17-2018.csv')
	return spending_process(school_data, scraped_data)

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
		'enrollment_by_grade': [
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
		],
		'endogeneous_input': [
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
			'Total In-district FTEs',
			'Total Expenditures',
			'Total Pupil FTEs',
			'Average Expenditures per Pupil'
		],
		'output_markers': [
			'District_Progress and Performance Index (PPI) - All Students'
		]
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
		'Less Than Highschool in Poverty',
		'Local government',
		'State government'
	]
	aux_morn = [
		'12am to 5am',
		'5am to 530am', 
		'530 am to 6am',
		'6am to 630am',
		'630 am to 7am',
		'7am to 730am',
		'730 am to 8am'
	]
	aux_eve = [
		'11am to 12noon',
		'12noon to 4pm',
		'4pm to midnight'
	]

	school_cols = school_categories['descriptive'] + school_categories['enrollment_by_grade'] + school_categories['endogeneous_input'] + school_categories['exogeneous_input'] + school_categories['output_markers']
	school_data = school_data[school_cols]
	school_data = school_data.rename(columns={'Zip':'Zip Code'})
	school_data = school_data.dropna()
	## Create Dummy Variables for School Type ##
	# one_hot_type = pd.get_dummies(school_data['School Type'])
	# school_data = school_data.join(one_hot_type)
	# school_data.drop('School Type', axis=1, inplace=True)
	#### Output Join Here ####

	zip_data['absent_morning'] = sum([zip_data[aux] for aux in aux_morn])
	zip_data['absent_evening'] = sum([zip_data[aux] for aux in aux_eve])
	zip_data = zip_data[zip_categories + ['Place', 'absent_morning', 'absent_evening']]

	zip_data = zip_data.rename(columns={'Place':'Zip Code'})
	## Join the zip code data with the school data ##
	joined = school_data.set_index('Zip Code').join(zip_data.set_index('Zip Code'), how='left', rsuffix='_scrape')
	
	## Normalization of numeric columns -- future work?##

	
	## Filter by school type -- bugs ##
	highschools = joined['12_Enrollment'] > 0
	middleschools = (joined['12_Enrollment'] == 0) & (joined['7_Enrollment'] > 0)
	elementaryschools = (joined['12_Enrollment'] == 0) & (joined['7_Enrollment'] == 0)
	
	## Filter the input and output columns
	x_cols = school_categories['endogeneous_input'] + school_categories['exogeneous_input'] + zip_categories # + ['Public School', 'Charter School']
	full_x = joined[x_cols]

	output = joined['District_Progress and Performance Index (PPI) - All Students']

	data_dict = {
		'full_x': full_x,
		'full_y': output,
		'highschool_x': full_x[highschools],
		'highschool_y': output[highschools],
		'middleschool_x': full_x[middleschools],
		'middleschool_y': output[middleschools],
		'elementary_x': full_x[elementaryschools],
		'elementary_y': output[elementaryschools]
	}

	return data_dict

def spending_process(school_data, zip_data):
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
		'enrollment_by_grade': [
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
		],
		'endogeneous_input': [
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
			'Total Expenditures',
			'Average Expenditures per Pupil'
		],
		'output_markers': [
			'Total # of Classes',
			'Average Class Size',
			'Salary Totals',
			'Average Salary',
			'FTE Count',
			'Total In-district FTEs',
			'Total Pupil FTEs'
		]
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
		'Less Than Highschool in Poverty',
		'Local government',
		'State government'
	]

	output_categories = [
			'Total # of Classes',
			'Average Class Size',
			'Salary Totals',
			'Average Salary',
			'FTE Count',
			'Total In-district FTEs',
			'Total Pupil FTEs'
	]
	aux_morn = [
		'12am to 5am',
		'5am to 530am', 
		'530 am to 6am',
		'6am to 630am',
		'630 am to 7am',
		'7am to 730am',
		'730 am to 8am'
	]
	aux_eve = [
		'11am to 12noon',
		'12noon to 4pm',
		'4pm to midnight'
	]

	school_cols = school_categories['descriptive'] + school_categories['enrollment_by_grade'] + school_categories['endogeneous_input'] + school_categories['exogeneous_input'] + school_categories['output_markers']
	school_data = school_data[school_cols]
	school_data = school_data.rename(columns={'Zip':'Zip Code'})
	school_data = school_data.dropna()
	## Create Dummy Variables for School Type ##
	# one_hot_type = pd.get_dummies(school_data['School Type'])
	# school_data = school_data.join(one_hot_type)
	# school_data.drop('School Type', axis=1, inplace=True)
	#### Output Join Here ####

	zip_data['absent_morning'] = sum([zip_data[aux] for aux in aux_morn])
	zip_data['absent_evening'] = sum([zip_data[aux] for aux in aux_eve])
	zip_data = zip_data[zip_categories + ['Place', 'absent_morning', 'absent_evening']]
	zip_data = zip_data[zip_categories + ['Place']]
	zip_data = zip_data.rename(columns={'Place':'Zip Code'})
	## Join the zip code data with the school data ##
	joined = school_data.set_index('Zip Code').join(zip_data.set_index('Zip Code'), how='left', rsuffix='_scrape')
	
	## Normalization of numeric columns -- future work?##

	
	## Filter by school type -- bugs ##
	highschools = joined['12_Enrollment'] > 0
	middleschools = (joined['12_Enrollment'] == 0) & (joined['7_Enrollment'] > 0)
	elementaryschools = (joined['12_Enrollment'] == 0) & (joined['7_Enrollment'] == 0)
	
	## Filter the input and output columns
	x_cols = school_categories['endogeneous_input'] + school_categories['exogeneous_input'] + zip_categories # + ['Public School', 'Charter School']
	full_x = joined[x_cols]

	output = joined[output_categories]

	data_dict = {
		'full_x': full_x,
		'full_y': output,
		'highschool_x': full_x[highschools],
		'highschool_y': output[highschools],
		'middleschool_x': full_x[middleschools],
		'middleschool_y': output[middleschools],
		'elementary_x': full_x[elementaryschools],
		'elementary_y': output[elementaryschools]
	}

	return data_dict

if __name__ == '__main__':
	main()