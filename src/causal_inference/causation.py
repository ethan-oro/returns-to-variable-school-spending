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

    predictions = predict_many(data, model, 50, 0.001)
    #plot_some(predictions)

    find_avg_slope(predictions)

def find_avg_slope(predictions):
    amnts_changed = predictions.keys()
    slopes = []
    for i in range(len(predictions[amnts_changed[0]])):
        outputs = [predictions[change][i] for change in amnts_changed]
        z = np.polyfit(amnts_changed, outputs, 1)
        p = np.poly1d(z)
        slopes.append(p.coeffs[0])
    print("mean: ")
    print(np.mean(slopes))
    print("standard deviation: ")
    print(np.std(slopes))

def predict_many(data, model, to_investigate, gap):
    predictions = {}

    current_avg_exp = data['highschool_x']['Average Expenditures per Pupil']
    current_exp = data['highschool_x']['Total Expenditures']
    for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
        data['highschool_x']['Average Expenditures per Pupil'] = change * current_avg_exp
        data['highschool_x']['Total Expenditures'] = change * current_exp
        predictions[change] = model.predict(data)

    return predictions

def plot_some(predictions, rows=3, cols=3):
    amnts_changed = predictions.keys()
    choices = np.random.choice(len(predictions[amnts_changed[0]]), rows * cols, replace=False)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True)

    subplots = axs.flatten()

    for i in range(len(choices)):
        choice = choices[i]
        ax = subplots[i]
        outputs = [predictions[change][choice] for change in amnts_changed]
        plot_one(ax, amnts_changed, outputs)

    plt.show()

def plot_one(ax, amnts_changed, outputs):
    ax.plot(amnts_changed, outputs, 'bo')
    ax.set_xlabel('Proportion of Current Spending', fontsize=7)
    ax.set_ylabel('Performance', fontsize=7)
    z = np.polyfit(amnts_changed, outputs, 1)
    p = np.poly1d(z)
    ax.plot(amnts_changed,p(amnts_changed),"r--")

if __name__ == '__main__':
    main()