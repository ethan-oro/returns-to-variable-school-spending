import sys, os
sys.path.append('./../data_processing/')
from dataprocess import *
from sklearn import linear_model
from sklearn import svm 
from sklearn.decomposition import PCA 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
Created by ksaac on 12/8/18
to do:
	1. take out the money related features
	2. run pca on the other features
	3. take the top 2 principal components 
	4. figure out how to do gradient colors for the y values
'''

def main():
	np.set_printoptions(precision=3)
	data = grab_data_full()
	spending_x = data['highschool_first_x']['Average Expenditures per Pupil']
	features_x = data['highschool_always_x']
	data_y = data['highschool_second_y']

	x, y,_,_ = transform_data(features_x, data_y, train_split = 1.0, standardize=True)
	
	pca_data = run_pca(x, 2)
	pca_data['Per-Student Spending'] = np.array(spending_x)
	plot3(pca_data, pca_data.columns, y,  "Plot of Spending and Principal Components \nvs. %s"%data_y.name)

def run_pca(data, num_components):
	'''
	run PCA to generate the |num_components| principal components of |data|
	
	@returns a matrix of principal components of same number of examples as |data|
	'''
	pca = PCA(n_components = num_components)
	principal_components = pca.fit_transform(data)
	explained_variance_pct = pca.explained_variance_ratio_
	column_names = ['Principal component %i: %.2f variance explained'%(i+1,var) for i,var in enumerate(explained_variance_pct)]
	component_data = pd.DataFrame(data = principal_components, columns = column_names)
	return component_data

def plot3(inputs, axis_names, targets, plot_title = 'title'):
	'''
	this function plots the data in 3d
	'''
	if len(axis_names) != 3:
		print(axis_names)
		print("Need 3d")

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	cb = ax.scatter(inputs[axis_names[0]], inputs[axis_names[1]], inputs[axis_names[2]],c=targets,cmap=plt.cm.get_cmap('RdBu'))
	ax.set_xlabel(axis_names[0])
	ax.set_ylabel(axis_names[1])
	ax.set_zlabel(axis_names[2])
	ax.set_title(plot_title)
	plt.colorbar(cb)
	plt.show()

def plot_model_results(data):
	return

if __name__ == "__main__":
	main()