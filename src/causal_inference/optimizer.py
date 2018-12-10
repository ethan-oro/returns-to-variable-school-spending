import sys, os
sys.path.append('./../data_processing/')
sys.path.append('./../models/')
from dataprocess import *
from models import *
from sklearn import linear_model
from sklearn import svm 
import matplotlib.pyplot as plt

def main():
    data = grab_data()
    model = Model(type='XGBoost with Bagging', regularization = False)
    model.train(data)

    optimize(data, model)



def predict_many(data, model, to_investigate, gap):
    predictions = {}

    current_avg_exp = data['highschool_x']['Average Expenditures per Pupil']
    current_exp = data['highschool_x']['Total Expenditures']
    for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
        data['highschool_x']['Average Expenditures per Pupil'] = change * current_avg_exp
        data['highschool_x']['Total Expenditures'] = change * current_exp
        predictions[change] = model.predict(data)

    return predictions


def optimize(data, model, penalty = 1.0, learning_rate = 0.01):
    ## set up the unconstrained minimization problem
    ## minimize -sum w_i*f(x_i) + penalty*||w_i dot x_i - B||
    def compute_objective(data, model):
        predictions = model.predict(data)
        print(data['highschool_x'].shape, predictions.shape)
        weighted_res = np.array(data['highschool_x']['Number of Students']).dot(predictions)
        print(weighted_res)

    compute_objective(data, model)


if __name__ == '__main__':
    main()