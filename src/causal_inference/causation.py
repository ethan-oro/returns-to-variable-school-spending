import sys, os
sys.path.append('./../data_processing/')
sys.path.append('./../models/')
from dataprocess import *
from models import *
from sklearn import linear_model
from sklearn import svm 
import matplotlib.pyplot as plt

def main():
    # data = grab_data()
    # model = Model(type='XGBoost with Bagging', regularization = False)
    # model.train(data)

    variators = ['Average Expenditures per Pupil', 'Average Salary', 'Average Class Size']
    output_metrics = ['Composite MCAS CPI', 'Composite SAT', '% Graduated', '% Attending College']
    predictions = {}
    for variator in variators:
        predictions[variator] = {}

    for output_metric in output_metrics:
        print('Running metric', output_metric)
        data = grab_data(output_metric)
        model = Model(type="linear_regression", regularization=False)
        errors = model.train(data)
        print (errors)
        for variator in variators:
            sampled_predictions = predict_many_2(data, model, variator, 50, 0.001)
            # plot_some(sampled_predictions)
            # break
            predictions[variator][output_metric] = sampled_predictions

    
    plot_all(predictions)

    # predictions = predict_many(data, model, 50, 0.001, 400)
    # plot_some(predictions)

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


def predict_many_2(data, model, variator, num_steps, step_size):
    predictions = {}
    current_var = data['highschool_x'][variator]
    for change in [1 + step_size * i for i in range(-num_steps, num_steps + 1)]:
        data['highschool_x'][variator] = change * current_var
        predictions[change] = model.predict(data)

    return predictions

def predict_many(data, model, to_investigate, gap, num_iters):
    predictions = {}

    current_avg_exp = data['highschool_x']['Average Expenditures per Pupil']
    current_exp = data['highschool_x']['Total Expenditures']
    for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
        predictions[change] = 0

    for _ in range(num_iters):
        model.train(data)
        for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
            data['highschool_x']['Average Expenditures per Pupil'] = change * current_avg_exp
            data['highschool_x']['Total Expenditures'] = change * current_exp
            predictions[change] += model.predict(data)

    for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
        predictions[change] /= num_iters

    return predictions

def plot_all(predictions, school_ind = 100):
    ncol = len(predictions[list(predictions.keys())[0]])
    fig, axs = plt.subplots(nrows=len(predictions.keys()), ncols=ncol, constrained_layout=True)
    subplots = axs.flatten()
    for i, (variator, outcome_dict) in enumerate(predictions.items()):
        for j, (output_metric, pred) in enumerate(outcome_dict.items()):
            amnts_changed = list(pred.keys())
            ax = subplots[i*ncol + j]
            outputs = [pred[change][school_ind] for change in amnts_changed]
            ax.set_title("%s \nvs. %s"%(variator, output_metric))
            ax.set_xlabel(variator, fontsize=7)
            ax.set_ylabel(output_metric, fontsize=7)
            plot_one(ax, amnts_changed, outputs)
            print(amnts_changed, outputs)

    plt.show()

def plot_some(predictions, rows=3, cols=3):
    amnts_changed = list(predictions.keys())

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
    ax.plot(amnts_changed, outputs, 'bx')

    z = np.polyfit(amnts_changed, outputs, 1)
    p = np.poly1d(z)
    ax.plot(amnts_changed,p(amnts_changed),"r--")

if __name__ == '__main__':
    main()