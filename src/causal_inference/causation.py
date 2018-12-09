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
	model = Model(type="linear_regression", regularization = False)
	model.train(data)

	predictions = predict_many(data, model, 25, 0.004)


def predict_many(data, model, to_investigate, gap):
	predictions = {}

	current_avg_exp = data['highschool_x']['Average Expenditures per Pupil']
	current_exp = data['highschool_x']['Total Expenditures']
	for change in [1 + gap * i for i in range(-to_investigate, to_investigate + 1)]:
		data['highschool_x']['Average Expenditures per Pupil'] = change * current_avg_exp
		data['highschool_x']['Total Expenditures'] = change * current_exp
		predictions[change] = model.predict(data)

	plot_some(predictions)

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
    ax.set_xlabel('Proportion of Current Spending', fontsize=6)
    ax.set_ylabel('Performance', fontsize=6)
    z = np.polyfit(amnts_changed, outputs, 1)
    p = np.poly1d(z)
    ax.plot(amnts_changed,p(amnts_changed),"r--")
    #ax.set_title('Title', fontsize=14)

if __name__ == '__main__':
	main()