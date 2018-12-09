## Using pandas to procees illinois school data into a more readable format

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
    print ('loading illinois...')
    data = load_data('../../data/illinois-public-schools-data/RC17_layout_csv.csv', '../../data/illinois-public-schools-data/rc17_assessment.txt')

def load_data(labels_files, data_files):
    df = pd.read_csv(data_files, delimiter=';')

if __name__ == '__main__':
    main()
