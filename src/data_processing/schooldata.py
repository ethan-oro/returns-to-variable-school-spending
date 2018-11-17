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
	print (df['Grade'])
	return df 

def process(frame):
	'''
	
	'''









if __name__ == '__main__':
	main()