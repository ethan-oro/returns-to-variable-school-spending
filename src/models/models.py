import sys, os
# import plotly.plotly as py
# import plotly.graph_objs as go
sys.path.append('./../data_processing/')
from dataprocess import *
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import discriminant_analysis
from sklearn import gaussian_process
import matplotlib.pyplot as plt
NUM_TRIALS = 100

def main():
    model_list = [ "linear", 'ridge', 'XGBoost', 'BaggingRegressor', 'RandomForest', 'Lasso', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost with Bagging', "Gaussian Process"]

    n_estimators=100
    subsample=1.0

    results = collections.defaultdict(tuple)
    for model in model_list:
        perform_model = Model(type=model, n_estimators=n_estimators, subsample=subsample)
        data_second = grab_data()
        avg_score_train, avg_score_test = multiple_splits(perform_model, data_second)
        results[model] = (avg_score_train, avg_score_test)
    for name, result in results.items():
        print('-- ', name ,' --')
        print('Average Training Score: ' + str(result[0]))
        print('Average Testng Score: ' + str(result[1]))


    N = len(model_list)

    train = []
    test = []

    for name, result in results.items():
        train.append(result[0])
        test.append(result[1])

    fig, ax = plt.subplots()

    ind = np.arange(N)    # the x locations for the groups
    width = 0.35         # the width of the bars
    p1 = ax.bar(ind, tuple(train), width, color='lightsteelblue')

    p2 = ax.bar(ind + width, tuple(test), width, color='c')

    ax.set_title('Model Performance with High School Graudation Output Label')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tuple(model_list))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_ylabel('R^2')

    ax.legend((p1[0], p2[0]), ('Train', 'Test'))
    ax.autoscale_view()

    plt.show()


def multiple_splits(model, data, noisy = False):
    sum_score_train = 0
    sum_score_test = 0
    for i in range(NUM_TRIALS):
        score_train, score_test = model.train(data)
        sum_score_train += score_train
        sum_score_test += score_test
        if noisy:
            print('---')
            print(score_train)
            print(score_test)
    avg_score_train = sum_score_train / NUM_TRIALS
    avg_score_test = sum_score_test / NUM_TRIALS

    return avg_score_train, avg_score_test

class Model(object):
    def __init__(self, type = "linear_regression", regularization = False, n_estimators = 100, subsample = 1.0):
        if type == "linear_regression":
            self.model = linear_model.LinearRegression(normalize=True)
        elif type == "ridge":
            self.model = linear_model.Ridge()
        elif type == "SVM":
            self.model = svm.SVR()
        elif type == 'XGBoost':
            self.model = ensemble.GradientBoostingRegressor(n_estimators=n_estimators, subsample=subsample)
        elif type == 'BaggingRegressor':
            self.model = ensemble.BaggingRegressor()
        elif type == 'RandomForest':
            self.model = ensemble.RandomForestRegressor()
        elif type == "AdaBoostRegressor":
            self.model = ensemble.AdaBoostRegressor()
        elif type == 'ExtraTreesRegressor':
            self.model = ensemble.ExtraTreesRegressor()
        elif type == 'Lasso':
            self.model = linear_model.Lasso()
        elif type == "qda":
            self.model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        elif type == "lda":
            self.model = discriminant_analysis.LinearDiscriminantAnalysis()
        elif type == 'XGBoost with Bagging':
            self.model = ensemble.BaggingRegressor(base_estimator=ensemble.GradientBoostingRegressor())
        elif type == "Gaussian Process":
            self.model = gaussian_process.GaussianProcessRegressor()


    def _transform_data(self, dataframe_x, dataframe_y, train_split = 0.8):
        m,n = dataframe_x.shape

        x = np.array(dataframe_x)
        y = np.array(dataframe_y)
        random_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(random_state)
        np.random.shuffle(y)
        split_ind = int(train_split*m)
        x_train = x[:split_ind,:]
        y_train = y[:split_ind]
        x_test = x[split_ind:,:]
        y_test = y[split_ind:]

        means = np.nanmean(x_train, axis=0)
        x_train = np.nan_to_num(x_train)
        x_test = np.nan_to_num(x_test)

        bad_inds_train = np.where(x_train == 0)
        bad_inds_test = np.where(x_test == 0)

        x_train[bad_inds_train] = np.take(means, bad_inds_train[1])
        x_test[bad_inds_test] = np.take(means, bad_inds_test[1])

        return (x_train, y_train, x_test, y_test)

    def _transform_data_pred(self, dataframe_x, dataframe_y, train_split = 0.8):
        m,n = dataframe_x.shape

        x = np.array(dataframe_x)
        y = np.array(dataframe_y)

        means = np.nanmean(x, axis=0)
        x = np.nan_to_num(x)

        bad_inds = np.where(x == 0)

        x[bad_inds] = np.take(means, bad_inds[1])

        return (x, y)

    def train(self, data, data_key = 'highschool', noisy = False):
        # data_key is one of 'full', 'highschool', 'middleschool', 'elementary'
        data_x = data['%s_x'%data_key]
        data_y = data['%s_y'%data_key]
        self.x_train, self.y_train, self.x_test, self.y_test = self._transform_data(data_x, data_y)
        self.model.fit(self.x_train, self.y_train)
        if noisy:
            print(self.model.predict(self.x_test))
            print('--')
            print(self.y_test)
            print('--')
            print(self.model.predict(self.x_test) - self.y_test)
        score_train = self.model.score(self.x_train, self.y_train)
        score_test = self.model.score(self.x_test, self.y_test)

        return score_train, score_test

    def predict(self, data, data_key = 'highschool'):
        data_x = data['%s_x'%data_key]
        data_y = data['%s_y'%data_key]
        x, y = self._transform_data_pred(data_x, data_y)
        return self.model.predict(x)



def old_spend():
    print('-- PART I --')
    data_first = grab_data_spend()
    for key in data_first['full_y'].keys():
        print(key)
        spend_model = Model(type="XGBoost", regularization = False)

        data_first = grab_data_spend()
        data_new = data_first
        data_new['full_y'] = data_first['full_y'][key]

        avg_score_train, avg_score_test = multiple_splits(spend_model, data_new)
        print('Average Training Score: ' + str(avg_score_train))
        print('Average Testng Score: ' + str(avg_score_test))

    perform_model = Model(type='full')
    data_second = grab_data()
    avg_score_train, avg_score_test = multiple_splits(perform_model, data_second)
    print('-- RESULTS --')
    print('Average Training Score: ' + str(avg_score_train))
    print('Average Testng Score: ' + str(avg_score_test))

def tuning():
    numIters = 20
    n_estimators=100
    subsample=1.0
    params = []
    for iter in range(0, numIters):
        perform_model = Model(type='XGBoost', n_estimators=n_estimators, subsample=subsample)
        data_second = grab_data()
        avg_score_train, avg_score_test = multiple_splits(perform_model, data_second)
        # print(perform_model.model.estimators_.shape)
        # tree = perform_model.model.estimators_[0, 0].tree_
        # leaf_mask = tree.children_left == -1
        # w_i = tree.value[leaf_mask, 0, 0]
        params.append(perform_model.model.estimators_)
        print (perform_model.model.estimators_.shape)
        print (type(perform_model.model.estimators_))
        print (type(perform_model.model.estimators_[0, 0].tree_))

    # estimators = []
    # for param in params:

    model = Model(type='XGBoost', n_estimators=n_estimators, subsample=subsample)
    # model.set_params()
    sum_score_train = 0
    sum_score_test = 0
    for i in range(NUM_TRIALS):
        score_train, score_test = model.train(data)
        sum_score_train += score_train
        sum_score_test += score_test
        if noisy:
            print('---')
            print(score_train)
            print(score_test)
    avg_score_train = sum_score_train / NUM_TRIALS
    avg_score_test = sum_score_test / NUM_TRIALS

    print('-- RESULTS --')
    print('Average Training Score: ' + str(avg_score_train))
    print('Average Testng Score: ' + str(avg_score_test))



    for n_estimators in range(60, 100, 10):
        for subsample in [0.6, 0.7, 0.8, 0.9, 1.0]:
            perform_model = Model(type='XGBoost', n_estimators=n_estimators, subsample=subsample)
            data_second = grab_data()

            avg_score_train, avg_score_test = multiple_splits(perform_model, data_second)
            print('-- ', models[0] , ': n_estimators: ', n_estimators, '; subsample: ',subsample, ' --')
            print('Average Training Score: ' + str(avg_score_train))
            print('Average Testng Score: ' + str(avg_score_test))



# def old_plotting():
#     test_score = np.zeros((n_estimators,), dtype=np.float64)

#     clf = perform_model.model
#     for i, y_pred in enumerate(clf.staged_predict(perform_model.x_test)):
#         test_score[i] = clf.loss_(perform_model.y_test, y_pred)


#     train = go.Scatter(x=np.arange(n_estimators) + 1,
#                        y=clf.train_score_,
#                        name='Training Set Deviance',
#                        mode='lines',
#                        line=dict(color='blue')
#                       )
#     test = go.Scatter(x=np.arange(n_estimators) + 1,
#                       y=test_score,
#                       mode='lines',
#                       name='Test Set Deviance',
#                       line=dict(color='red')
#                      )

#     layout = go.Layout(title='Deviance',
#                        xaxis=dict(title='Boosting Iterations'),
#                        yaxis=dict(title='Deviance')
#                       )
#     fig = go.Figure(data=[test, train], layout=layout)
#     pio.write_image(fig, 'fig1.png')

#     print('finished plotting')





if __name__ == '__main__':
    main()
