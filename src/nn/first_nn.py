#first crack at deep learning

# sys imports
import sys, os, shutil, errno

# string/data inputs
import string, csv, json, fileinput, math

# misc
import progressbar
import collections
import pickle #for saving data objects

# gucci imports
import numpy as np
import pandas as pd
import sklearn as skl
from keras.models import Sequential
from keras.layers import Dense

# data imports
sys.path.insert(0, '../data_processing')
from dataprocess import school_process, load_csv

#i split them up like this so we can use scikit analysis later
def run_nn():
    train_x, train_y, test_x, test_y = grab_data()
    model = build_model(60)
    trained_model = train(model, train_x, train_y)
    preds = test(trained_model, test_x)
    compare(preds, test_y)

#grabs the data we want and splits it train / test
def grab_data():
    school_data = load_csv('../../data/massachusetts-public-schools-data/MA_Public_Schools_2017.csv')
    scraped_data = load_csv('../../scraper/econ_full_scrape_11-17-2018.csv')
    data = school_process(school_data, scraped_data)
    hs_x = data['highschool_x']
    hs_y = data['highschool_y']
    bad_x = [ index for index, row in hs_x.iterrows() if row.isnull().any() == True ]
    bad_y = [ index for index, row in hs_y.iteritems() if math.isnan(row) ]
    print ('x', bad_x)
    print ('y', bad_y)

    bad = list(set(bad_x + bad_y))
    hs_x.drop(bad)
    hs_y.drop(bad)

    print ('----')
    print('x', hs_x.isnull().any())
    print('y', hs_x.isnull().any())
    split = (int)(len(hs_x) * 0.8)
    # print(data['highschool_x'].dtypes)
    # print(data['highschool_x'].values)
    # c = ''
    # for a in data['highschool_x'].values:
    #     print (type(a))
    #     for b in a:
    #         c += str(type(b))
    #     print (c)
    #     c = ''
    # for a in data['highschool_x'].values[:split]:
    #     print (type(a))
    #     for b in a:
    #         c += str(type(b))
    #     print (c)
    #     c = '

    model = Sequential()
    model.add(Dense(30, input_dim=60, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    fitted = model.fit(hs_x.values[:split], hs_y.values[:split], epochs=500, verbose=1)


    # x_t = [ elem for elem in data['highschool_x'].iloc[:split].values ]
    # b = ''
    # for elem in x_t[:5]:
    #     a = type(elem)
    #     for e in elem:
    #         b += str(type(elem))
    #     print (a, b)
    #     print (elem)
    #     b = ''
    # print (x_t)

    # return hs_x[:split], hs_y[:split], hs_x[split:], hs_y[split:]


#basic sequential linreg model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(120, input_dim=input_dim, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

def train(model, train_x, train_y):
    b = ''
    for elem in train_x:
        a = type(elem)
        for e in elem:
            b += str(type(elem))
        print (a, b)
        break
    return model.fit(train_x, train_y, epochs=500, verbose=1)

def test(model, test_x):
    return model.predict(test_x)

def compare(preds, test_y):
    print(np.mean((pred - test_y) ** 2, axis = 0))





if __name__ == '__main__':
    # run_nn()
    grab_data()
