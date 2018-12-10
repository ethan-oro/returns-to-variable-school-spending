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
    data = load_data('../../data/illinois-public-schools-data/new_headers.csv', '../../data/illinois-public-schools-data/rc17_assessment.txt')

def load_data(labels_files, data_files):
    df_labels = pd.read_csv(labels_files, sep =',', encoding='latin-1')
    labels1 = df_labels['Desired1']
    labels1 = labels1.dropna().tolist()
    labels2 = df_labels['Desired2']
    labels2 = labels2.dropna().tolist()
    print(len(labels1))
    print(len(labels2))
    labels = ["{} {}".format(b_, a_) for a_, b_ in zip(labels1, labels2)]
    # for i in range(len(labels)):
    #     for j in range(i + 1, len(labels)):
    #         if (labels[i] == labels[j]):
    #             print (labels[i], i, j)
    labels.append('NaaN')
    df = pd.read_csv(data_files, sep=';', header=None)
    print (df)
    df.columns = labels
    print (df[labels[4]])
    print (df[labels[3]])

if __name__ == '__main__':
    main()
