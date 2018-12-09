import sys, os
sys.path.append('./../data_processing/')
from dataprocess import *
from sklearn import linear_model
from sklearn import svm 
NUM_TRIALS = 100

def main():
	print('-- PART I --')
	data_first = grab_data_spend()
	for key in data_first['full_y'].keys():
		print(key)
		spend_model = Model(type="linear_regression", regularization = True)

		data_first = grab_data_spend()
		data_new = data_first
		data_new['full_y'] = data_first['full_y'][key]

		avg_score_train, avg_score_test = multiple_splits(spend_model, data_new)
		print('Average Training Score: ' + str(avg_score_train))
		print('Average Testng Score: ' + str(avg_score_test))


	perform_model = Model(type="linear_regression", regularization = False)
	data_second = grab_data()

	avg_score_train, avg_score_test = multiple_splits(perform_model, data_second)
	print('-- PART II --')
	print('Average Training Score: ' + str(avg_score_train))
	print('Average Testng Score: ' + str(avg_score_test))



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
	def __init__(self, type = "linear_regression", regularization = False):
		if type == "linear_regression":
			if regularization:
				self.model = linear_model.Ridge()
			else:
				self.model = linear_model.LinearRegression(normalize=False)
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

	def predict(self, data, data_key = 'highschool', noisy = False):
		data_x = data['%s_x'%data_key]
		data_y = data['%s_y'%data_key]
		x_train, y_train, x_test, y_test = self._transform_data(data_x, data_y)
		return self.model.predict(x_test)






if __name__ == '__main__':
	main()