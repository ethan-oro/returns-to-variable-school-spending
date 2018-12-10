import sys, os
sys.path.append('./../data_processing/')
sys.path.append('./../models/')
from dataprocess import *
from models import *
from sklearn import linear_model
from sklearn import svm 
import matplotlib.pyplot as plt
import copy

def main():
    data = grab_data()
    model = Model(type='XGBoost with Bagging', regularization = False)
    model.train(data)

    optimize(data, model)



def find_avg_slope(predictions):
    amnts_changed = predictions.keys()
    slopes = []
    for i in range(len(predictions[amnts_changed[0]])):
        outputs = [predictions[change][i] for change in amnts_changed]
        z = np.polyfit(amnts_changed, outputs, 1)
        p = np.poly1d(z)
        slopes.append(p.coeffs[0])
    
    return np.mean(slopes)

def predict_many(data, model, to_investigate, gap):
    predictions = {}

    current_avg_exp = data['highschool_x']['Average Expenditures per Pupil']
    current_exp = data['highschool_x']['Total Expenditures']
    for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
        data['highschool_x']['Average Expenditures per Pupil'] = change * current_avg_exp
        predictions[change] = model.predict(data)

    return predictions


def optimize(data, model, penalty = 0.003, learning_rate = 0.001):
    ## set up the unconstrained minimization problem
    ## minimize -sum w_i*f(x_i) + penalty*||w_i dot x_i - B||
    def compute_objective(data, model):
        predictions = model.predict(data)
        weighted_res = np.array(data['highschool_x']['Number of Students']).dot(predictions)
        return weighted_res

    def numerical_derivative(data, model, a = 0.1):
        data_aug = copy.deepcopy(data) #create a copy of the data
        spend_aug = np.array(data_aug['highschool_x']['Average Expenditures per Pupil']) + a
        data_aug['highschool_x']['Average Expenditures per Pupil'] = spend_aug
        # print(data['highschool_x']['Average Expenditures per Pupil'] - data_aug['highschool_x']['Average Expenditures per Pupil'])
        
        fx = model.predict(data)
        fx_a = model.predict(data_aug)
        deriv = (fx_a - fx) / a
        return deriv

    original_spending = np.array(data['highschool_x']['Number of Students']).dot(data['highschool_x']['Average Expenditures per Pupil'])
    
    for i in range(10000):
        
        deriv = numerical_derivative(data, model, 1000)
        scaled_deriv = data['highschool_x']['Number of Students']*deriv
        new_spending = np.array(data['highschool_x']['Number of Students']).dot(data['highschool_x']['Average Expenditures per Pupil'])
        constraint_penalizer = 2*penalty*(new_spending - original_spending)
        # print(original_spending, new_spending)
        if (i % 100 == 0):
            print "Objective", compute_objective(data, model), new_spending, constraint_penalizer
        new_x = data['highschool_x']['Average Expenditures per Pupil'] - learning_rate*(-scaled_deriv + constraint_penalizer)
        data['highschool_x']['Average Expenditures per Pupil'] = new_x

if __name__ == '__main__':
    main()