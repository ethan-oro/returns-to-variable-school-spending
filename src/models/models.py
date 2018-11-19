import sys, os
sys.path.append('./../data_processing/')
from dataprocess import *
from sklearn import linear_model
from sklearn import svm 


def main():
	model = Model()
	model.train()

class Model(object):
	def __init__(self, type = "linear_regression", regularization = False):
		if type == "linear_regression":
			if regularization:
				self.model = linear_model.Ridge()
			else:
				self.model = linear_model.LinearRegression(normalize=True)
		elif type == "SVM":
			self.model = svm.SVR()

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

		# std = x_train.std(axis=0)
		
		# means = means.reshape((n,1)).T
		# std = std.reshape((n,1)).T
		
		# x_train = (x_train - means) / std 
		# x_test = (x_test - means) / std

		return (x_train, y_train, x_test, y_test)

	def train(self, data_key = 'highschool'):
		# data_key is one of 'full', 'highschool', 'middleschool', 'elementary'
		data = grab_data()

		data_x = data['%s_x'%data_key]
		data_y = data['%s_y'%data_key]
		self.x_train, self.y_train, self.x_test, self.y_test = self._transform_data(data_x, data_y)
		self.model.fit(self.x_train, self.y_train)
		print(self.model.predict(self.x_test))
		print('--')
		print(self.y_test)
		print('--')
		print(self.model.predict(self.x_test) - self.y_test)
		print(self.model.score(self.x_train, self.y_train))
		print(self.model.score(self.x_test, self.y_test))






if __name__ == '__main__':
	main()