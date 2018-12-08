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
from keras import regularizers
from sklearn.metrics import r2_score

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
    #gets data
    school_data = load_csv('../../data/massachusetts-public-schools-data/MA_Public_Schools_2017.csv')
    scraped_data = load_csv('../../scraper/econ_full_scrape_11-17-2018.csv')
    data = school_process(school_data, scraped_data)
    dataframe_x = data['highschool_x']
    dataframe_y = data['highschool_y']
    m,n = dataframe_x.shape
    bad_x = [ index for index, row in dataframe_x.iterrows() if row.isnull().any() == True ]
    bad_y = [ index for index, row in dataframe_y.iteritems() if math.isnan(row) ]
    bad = list(set(bad_x + bad_y))
    dataframe_x = dataframe_x.drop(bad)
    dataframe_y = dataframe_y.drop(bad)
    x = np.array(dataframe_x)
    y = np.array(dataframe_y)
    random_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(random_state)
    np.random.shuffle(y)
    m,n = x.shape
    split_ind = int(0.8*m)
    x_train = x[:split_ind,:]
    y_train = y[:split_ind]
    x_test = x[split_ind:,:]
    y_test = y[split_ind:]

    print (x_train)
    print (y_train)
    print (x_test)
    print (y_test)

    #lame loop to remove one stray element
    for x in x_test:
        for i in range(len(x)):
            if (x[i] == '<a href=\"2651-Zipcode-MA-Economy-data.html\">2651</a>'):
                print ("yes")
                x[i] = 0.0
    for x in x_train:
        for i in range(len(x)):
            if (x[i] == '<a href=\"2651-Zipcode-MA-Economy-data.html\">2651</a>'):
                print ("yes")
                x[i] = 0.0

    # for x in x_test:
    #     for i in range(len(x)):
    #         if (x[i] == '<a href=\"2651-Zipcode-MA-Economy-data.html\">2651</a>'):
    #             print ("yes")
    #             x[i] = 0.0
    #
    # means = np.nanmean(x_train, axis=0)
    # x_train = np.nan_to_num(x_train)
    # x_test = np.nan_to_num(x_test)
    #
    # bad_inds_train = np.where(x_train == 0)
    # bad_inds_test = np.where(x_test == 0)
    #
    # x_train[bad_inds_train] = np.take(means, bad_inds_train[1])
    # x_test[bad_inds_test] = np.take(means, bad_inds_test[1])


    # #cleans data for null values
    # hs_x = data['highschool_x']
    # hs_y = data['highschool_y']
    # bad_x = [ index for index, row in hs_x.iterrows() if row.isnull().any() == True ]
    # bad_y = [ index for index, row in hs_y.iteritems() if math.isnan(row) ]
    # bad = list(set(bad_x + bad_y))
    # hs_x = hs_x.drop(bad)
    # hs_y = hs_y.drop(bad)
    #
    # #splits data into train/test on an 80/20 ratio
    # split = (int)(len(hs_x) * 0.8)
    # x_train = hs_x.values[:split]
    # y_train = hs_y.values[:split]
    # x_test = hs_x.values[split:]
    # y_test = hs_y.values[split:]
    #
    # #lame loop to remove one stray element
    # for x in x_test:
    #     for i in range(len(x)):
    #         if (x[i] == '<a href=\"2651-Zipcode-MA-Economy-data.html\">2651</a>'):
    #             print ("yes")
    #             x[i] = 0.0

    #intits model
    model = Sequential()
    model.add(Dense(30, input_dim=60, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    #trains model
    model.fit(x_train, y_train, epochs=500, verbose=0)

    #make and score predictions
    preds = model.predict(x_test)
    print(y_test)
    print (preds)
    print(np.mean((preds - y_test) ** 2, axis = 0))
    print(r2_score(preds, y_test))

    # avg_error = {}
    # for first in range(5, 54)[0::5]:
    #     for second in range(5, 59 - first)[0::5]:
    #         print (first, second)
    #         #intits model
    #         model = Sequential()
    #         model.add(Dense(first, input_dim=60, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    #         model.add(Dense(second, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    #         model.add(Dense(1, activation='linear'))
    #         model.compile(loss='mse', optimizer='adam')
    #
    #         #trains model
    #         model.fit(train_x, train_y, epochs=200, batch_size=10, verbose=0)
    #
    #         #make and score predictions
    #         preds = model.predict(test_x)
    #         avg_error[(first, second)] = (sum(np.mean((preds - test_y) ** 2, axis = 0)), r2_score(preds, test_y))
    # min_mse = (sys.float_info.max,)
    # max_r = (sys.float_info.min,)
    # for k, v in avg_error.items():
    #     print (k, ' : ' , v)
    #     if (v[0] < min_mse[0]):
    #         min_mse = (v[0], k)
    #     if (v[1] > max_r[0]):
    #         max_r = (v[1], k)
    # print ('best mse was : ', min_mse)
    # print ('best r was : ', max_r)
#basic sequential linreg model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(120, input_dim=input_dim, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

def train(model, train_x, train_y):
    return model.fit(train_x, train_y, epochs=500, verbose=1)

def test(model, test_x):
    return model.predict(test_x)

def compare(preds, test_y):
    print(np.mean((pred - test_y) ** 2, axis = 0))





if __name__ == '__main__':
    # run_nn()
    grab_data()
