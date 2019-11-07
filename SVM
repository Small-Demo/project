import random
import numpy as np
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers, spmatrix


def createData(size):
	traningdata = np.zeros((size, 3))
	random.seed(10)
	for i in range(size):
		while True:
			traningdata[i, 0] = random.uniform(0, 100)
			traningdata[i, 1] = random.uniform(0, 100)
			# f(x)=0.5x+20 d=10
			if 2 * traningdata[i, 1] - traningdata[i, 0] > 50:
				traningdata[i, 2] = 1
				break
			elif 2 * traningdata[i, 1] - traningdata[i, 0] < 30:
				traningdata[i, 2] = -1
				break
	return traningdata


def addnoise(data):
	#for _ in range(12):
	#	r = random.randint(0, 99)
	#	data[r, 2] = - data[r, 2]

	data[0, :] = 40.0, 38.0, +1
	data[1, :] = 60.0, 50.5, -1
	return data


def hmqp_svm(data):
	size = data.shape[0]
	x1 = data[:, 0]
	x2 = data[:, 1]
	y = data[:, 2]
	P = spmatrix([1., 1.], [1, 2], [1, 2], (3, 3))
	q = matrix(np.zeros((3, 1)))
	g = np.empty([size, 3])
	g[:, 0] = -y
	g[:, 1] = np.multiply(-y, x1)
	g[:, 2] = np.multiply(-y, x2)
	G = matrix(g)
	h = -matrix(np.ones((size, 1)))
	sol = solvers.qp(P, q, G, h)
	return sol['x']


def hm_svm(data):
	size = data.shape[0]
	x = data[:, 0:2]
	y = data[:, 2]
	K = np.empty((size, size))
	for i in range(0, size):
		for j in range(0, size):
			K[i, j] = np.dot(x[i], x[j])
	P = matrix(np.multiply(np.outer(y, y), K))
	q = matrix(-np.ones((size, 1)))
	G = matrix(np.diag(-np.ones(size)))
	h = matrix(np.zeros((size, 1)))
	A = matrix(y, (1, size))
	b = matrix(0.0)
	sol = solvers.qp(P, q, G, h, A, b)
	alphas = np.ravel(sol['x'])
	w = np.dot(np.multiply(alphas, y), x)
	for i in range(size):
		if alphas[i] > 0:
			b = y[i] - np.dot(np.multiply(alphas, y), K[i])
			break
	return np.asarray([b, w[0], w[1]])


def sm_svm(data, c):
	size = data.shape[0]
	x = data[:, 0:2]
	y = data[:, 2]
	K = np.empty((size, size))
	for i in range(0, size):
		for j in range(0, size):
			K[i, j] = np.dot(x[i], x[j])
	P = matrix(np.multiply(np.outer(y, y), K))
	q = matrix(-np.ones((size, 1)))
	G = matrix(np.vstack((np.diag(-np.ones(size)), np.identity(size))))
	h = matrix(np.hstack((np.zeros(size), np.ones(size) * c)))
	A = matrix(y, (1, size))
	b = matrix(0.0)
	solvers.options["show_progress"] = False
	sol = solvers.qp(P, q, G, h, A, b)
	alphas = np.ravel(sol['x'])
	w = np.dot(np.multiply(alphas, y), x)
	for i in range(size):
		if 0 < alphas[i] < c:
			b = y[i] - np.dot(np.multiply(alphas, y), K[i])
			break
	return np.asarray([b, w[0], w[1]])


def cross_validation(data):
	# 5倍交叉验证选C
	size = data.shape[0]
	best_accuracy = 0
	for c in [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
		acc = []
		for i in range(5):
			s = np.vsplit(data, 5)
			test = s.pop(i)
			train = np.vstack((s[0], s[1], s[2], s[3]))
			w = sm_svm(train, c)
			x = np.hstack((np.ones((size // 5, 1)), test[:, 0:2]))
			h = np.sign(np.dot(x, w))
			acc.append(np.where(np.multiply(h, test[:, 2]) == 1)[0].size / (size // 5))
		accuracy = round(np.mean(np.asarray(acc)), 4)
		print("when C is {},the accuracy is {}".format(c, accuracy))
		if accuracy > best_accuracy:
			best_c = c
			best_accuracy = accuracy
	print("The best C is {}".format(best_c))
	return best_c


def draw(data, weight):
	size = data.shape[0]
	plt.title("Trainning Data")
	for i in range(size):
		if data[i, 2] == 1:
			plt.plot(data[i, 0], data[i, 1], "bo")
		else:
			plt.plot(data[i, 0], data[i, 1], "rx")
	plt.xlim((-10, 110))
	plt.ylim((-10, 110))
	plt.xlabel("Feature X1")
	plt.ylabel("Feature X2")
	# f(x)线
	x = np.asarray([0, 100])
	y = 0.5 * x + 20
	plt.plot(x, y, "k")
	# g(x)线
	z = -(weight[1]/weight[2]*x + weight[0]/weight[2])
	plt.plot(x, z, "g")
	plt.show()


if __name__ == "__main__":
	data = createData(100)
	# w = hm_svm(data)
	# data = addnoise(data)
	c = cross_validation(data)
	w = sm_svm(data, c)
	print(w)
	print(w[1]/w[2], w[0]/w[2])
	draw(data, w)
