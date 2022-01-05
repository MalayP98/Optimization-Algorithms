import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

matplotlib.use('TKAgg')
warnings.simplefilter(action='ignore', category=FutureWarning)

def linear_dataset(slope, intercept, size, noise):
    x = np.random.uniform(1, 10, size=size)
    y = x * slope + intercept + (noise * np.random.uniform(-2, 2, [1, size]))
    dataset = np.concatenate((x.reshape(size, 1), y.reshape(size, 1)), axis=1)
    return dataset

def train_test_split(input, target, ratio):
    shuffle_indices = np.random.permutation(input.shape[0])
    test_size = int(input.shape[0] * ratio)
    train_indices = shuffle_indices[:test_size]
    test_indices = shuffle_indices[test_size:]
    xTrain = input[train_indices]
    yTrain = target[train_indices]
    xTest = input[test_indices]
    yTest = target[test_indices]
    return xTrain, yTrain, xTest, yTest

def scale(data):
    for i in range(data.shape[1]):
        col_i = data[:, i]
        data[:, i] = (col_i - min(col_i)) / (max(col_i) - min(col_i))
    return data

class LinearRegression:

    def __init__(self, optimizer='GD', dynamic_learning_rate="None"):
        """
    3 kinds of learning rates:
 
    a) exponetially decreasing 
    b) polynomially decreasing
    c) Constant
    """
        self.history = None
        self.optimizer = optimizer
        self.dlr = dynamic_learning_rate

    def getHistory(self):
        return self.history

    def fit(self, X, Y):
        X = scale(X.reshape(len(X), 1))
        self.X = np.c_[X, np.ones((len(X), 1))]
        self.Y = Y.reshape(len(Y), 1)
        self.theta = np.random.rand(self.X.shape[1], 1)

    def exponential_decay(self, eta, epoch, decay):
        return eta * (np.exp(decay * epoch))

    def polynomial_decay(self, eta, epoch, beta, alpha):
        return eta * ((1 + beta * epoch) ** alpha)

    def get_stochastic_gradient(self, X, num_points, error):
        random_points = np.random.randint(0, len(X), num_points)
        new_data = []
        new_error = []
        for point in random_points:
            new_data.append(X[point])
            new_error.append(error[point])
        new_data = np.array(new_data)
        new_error = np.array(new_error)
        gradient = (2 / num_points) * np.dot(np.transpose(new_data), new_error)
        return gradient

    # Algorithm 1
    def GradientDecesent(self, epochs, eta):
        previous_theta = []
        previous_error = []
        for epoch in range(epochs):
            if self.dlr == "exponential": eta = self.exponential_decay(eta, epoch, -0.01)
            if self.dlr == "polynomial": eta = self.polynomial_decay(eta, epoch, 0.1, -0.5)
            predicted = np.dot(self.X, self.theta)
            error = predicted - self.Y
            gradient = (2 / len(self.X)) * np.dot(np.transpose(self.X), error)
            self.theta = self.theta - eta * gradient
            error = (np.dot(np.transpose(error), error)) / len(X)
            previous_theta.append(self.theta)
            previous_error.append(error[0])
        return [previous_theta, previous_error]

    # Algorithm 2
    def Momentum(self, epochs, eta, gamma, mini_batch=70):
        previous_theta = []
        previous_error = []
        momentum = np.array(len(self.theta) * [0])
        momentum = momentum.reshape(len(momentum), 1)
        for epoch in range(1, epochs + 1):
            if self.dlr == "exponential": eta = self.exponential_decay(eta, epoch, -0.01)
            if self.dlr == "polynomial": eta = self.polynomial_decay(eta, epoch, 0.1, -0.5)
            predicted = np.dot(self.X, self.theta)
            error = predicted - self.Y
            gradient = self.get_stochastic_gradient(self.X, mini_batch, error)
            momentum = gamma * momentum + (1 - gamma) * gradient
            self.theta = self.theta - eta * momentum
            error = (np.dot(np.transpose(error), error)) / len(X)
            previous_theta.append(self.theta)
            previous_error.append(error[0])
        return [previous_theta, previous_error]

    # Algorithm 3
    def RMSProp(self, epochs, eta, beta, mini_batch=70):
        previous_theta = []
        previous_error = []
        moving_avg = 0
        for epoch in range(1, epochs + 1):
            predicted = np.dot(self.X, self.theta)
            error = predicted - self.Y
            gradient = self.get_stochastic_gradient(self.X, mini_batch, error)
            moving_avg = beta * moving_avg + (1 - beta) * gradient**2
            self.theta = self.theta - ((eta * gradient) / (moving_avg ** 0.5))
            error = (np.dot(np.transpose(error), error)) / len(X)
            previous_theta.append(self.theta)
            previous_error.append(error[0])
        return [previous_theta, previous_error]

    # Alogrithm 4
    def Adagrad(self, epochs, eta, mini_batch=70):
        previous_theta = []
        previous_error = []
        moving_avg = 0
        for epoch in range(1, epochs+1):
            predicted = np.dot(self.X, self.theta)
            error = predicted - self.Y
            gradient = self.get_stochastic_gradient(self.X, mini_batch, error)
            moving_avg = moving_avg + gradient**2
            self.theta = self.theta - ((eta * gradient) / (moving_avg ** 0.5))
            error = (np.dot(np.transpose(error), error)) / len(X)
            previous_theta.append(self.theta)
            previous_error.append(error[0])
        return [previous_theta, previous_error]

    # Algorithm 5
    def Adam(self, epochs, eta, beta1, beta2, mini_batch=70):
        previous_theta = []
        previous_error = []
        moving_avg_1 = 0
        moving_avg_2 = 0
        for epoch in range(1, epochs+1):
            predicted = np.dot(self.X, self.theta)
            error = predicted - self.Y
            gradient = self.get_stochastic_gradient(self.X, mini_batch, error)
            moving_avg_1 = beta1 * moving_avg_1 + (1-beta1) * gradient
            moving_avg_2 = beta2 * moving_avg_2 + (1-beta2) * (gradient**2)
            mvavg1_hat = moving_avg_1 / (1 - np.power(beta1, epoch))
            mvavg2_hat = moving_avg_2 / (1 - np.power(beta2, epoch))
            self.theta = self.theta - eta * mvavg1_hat / (np.sqrt(mvavg2_hat) + 0.0001)
            error = (np.dot(np.transpose(error), error)) / len(X)
            previous_theta.append(self.theta)
            previous_error.append(error[0])
        return [previous_theta, previous_error]



    def train(self, epochs, eta, mini_bacth=70):
        self.epochs = epochs
        self.mini_bacth = mini_bacth
        if self.optimizer == "GD":
            self.history = self.GradientDecesent(epochs, eta)
        elif self.optimizer == "Momentum":
            self.history = self.Momentum(epochs, eta, 0.9, mini_bacth)
        elif self.optimizer == "RMSP":
            self.history = self.RMSProp(epochs, eta, 0.9)
        elif self.optimizer == "Adagrad":
            self.history = self.Adagrad(epochs, eta)
        elif self.optimizer == "Adam":
            self.history = self.Adam(epochs, eta, 0.9, 0.99)

    def predict(self, X, Y):
        X = scale(X.reshape(len(X), 1))
        X = np.c_[X, np.ones((len(X), 1))]
        Y = Y.reshape(len(Y), 1)
        predicted = np.dot(X, self.theta)
        net_error = predicted - Y
        net_error = (2 / len(X)) * np.dot(np.transpose(net_error), net_error)
        print("Error for {} = {}".format(self.optimizer, net_error[0][0]))
        return predicted

    def show_transition(self):
        previous_theta = self.history[0]
        for theta in previous_theta:
            predicted = np.dot(self.X, theta)
            plt.plot(self.X[:, 0], predicted)
        plt.scatter(self.X[:, 0], self.Y, alpha=0.7)
        plt.title("Hypothesis Transition")
        plt.show()

    def show_lossCurve(self):
        iteration = np.arange(self.epochs)
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(iteration.reshape(self.epochs, 1), self.history[1])
        plt.show()

    def show_weightTransition(self):
        theta1 = [row[0] for row in self.history[0]]
        bias = [row[1] for row in self.history[0]]
        epochs = np.arange(1, self.epochs + 1)
        plt.title("Weight Transition")
        plt.xlabel("Epochs")
        plt.ylabel("Weight")
        plt.plot(epochs, theta1, label="theta1")
        plt.plot(epochs, bias, label="bais")
        plt.legend(loc="upper left")
        plt.show()

    def final_fit(self):
        plt.scatter(self.X[:, 0], self.Y, alpha=0.7)
        predicted = np.dot(self.X, self.theta)
        plt.plot(self.X[:, 0], predicted)
        plt.show()

class Visualize:
    def __init__(self, loss_fun='MSE', features=None, plot_transition=False, history=None):
        self.features = features
        self.loss_fun = loss_fun
        self.plot_transition = plot_transition
        self.theta1 = None
        self.theta2 = None
        self.error = None
        self.history = history

    def mean_sqr_error(self, x, y, x_axis, y_axis):
        z_axis = [[0] * 100] * 100
        for i in range(0, 100):
            for j in range(0, 100):
                cordinates = [[x_axis[i][j]], [y_axis[i][j]]]
                predict = np.dot(x, cordinates)
                error = predict - y
                error = (2 / len(x)) * np.dot(np.transpose(error), error)
                z_axis[i][j] = error[0][0]
        z_axis = np.array(z_axis)
        return z_axis

    def plot_type(self, type_=None, res=100, hzr_angle=None, ver_angle=None, x_axis=None, y_axis=None, z_axis=None):
        if type_ == 'contour':
            plt.scatter(x_axis, y_axis, c=z_axis)
            plt.show()

        elif type_ == '3d':
            ax = plt.axes(projection='3d')
            fig = plt.figure()
            ax.contour3D(x_axis, y_axis, z_axis, res)
            ax.view_init(hzr_angle, ver_angle)

    def plot(self, type_=None, res=5000, hzr_angle=None, ver_angle=None):
        x = np.arange(-1, 1, 0.1)
        x = np.c_[x, np.ones((len(x), 1))]
        y = np.dot(x, features)
        dims = []
        for _ in range(0, 2):
            dims.append(np.arange(-12, 12, 0.24))
        x_axis, y_axis = np.meshgrid(dims[0], dims[1])

        if self.loss_fun == 'MSE':
            z_axis = self.mean_sqr_error(x, y, x_axis, y_axis)
            if self.plot_transition:
                weight = self.history[0]
                error = self.history[1]
                weight = np.array(weight).reshape(len(weight), 2)
                theta1 = []
                theta2 = []
                for cordi in weight:
                    theta1.append(cordi[0])
                    theta2.append(cordi[1])
                error = np.array(error).reshape(len(error), )
                self.theta1 = theta1
                self.theta2 = theta2
                self.error = error
                ax = plt.axes(projection='3d')
                ax.scatter3D(self.theta1, self.theta2, self.error)
                return
            self.plot_type(type_, res, hzr_angle, ver_angle, x_axis, y_axis, z_axis)

ITERATIONS = 100

dataset = linear_dataset(2, 3, 300, 4.5)
features = np.array([[2], [3]])
X = dataset[:, 0]
Y = dataset[:, 1]
X_train, y_train, X_test, y_test = train_test_split(X, Y, .3)

#Gadient Descent
obj = LinearRegression("GD")
obj.fit(X_train, y_train)
obj.train(ITERATIONS, 0.1)
obj.show_transition()
obj.show_lossCurve()
historyGD = obj.getHistory()
obj.show_weightTransition()
obj.final_fit()
predicted = obj.predict(X_test, y_test)

#Momentum
obj = LinearRegression("Momentum")
obj.fit(X_train, y_train)
obj.train(ITERATIONS, 0.2, 70)
obj.show_transition()
obj.show_lossCurve()
historyMomentum = obj.getHistory()
obj.show_weightTransition()
obj.final_fit()
predicted = obj.predict(X_test, y_test)

#RMSP
obj = LinearRegression("RMSP")
obj.fit(X_train, y_train)
obj.train(ITERATIONS, 0.2)
obj.show_transition()
obj.show_lossCurve()
historyRMSP = obj.getHistory()
obj.show_weightTransition()
obj.final_fit()
predicted = obj.predict(X_test, y_test)

#Adagrad
obj = LinearRegression("Adagrad")
obj.fit(X_train, y_train)
obj.train(ITERATIONS, 0.2)
obj.show_transition()
obj.show_lossCurve()
historyAda = obj.getHistory()
obj.show_weightTransition()
obj.final_fit()
predicted = obj.predict(X_test, y_test)

#Adam
obj = LinearRegression("Adam")
obj.fit(X_train, y_train)
obj.train(ITERATIONS, 2)
obj.show_transition()
obj.show_lossCurve()
historyAdam = obj.getHistory()
obj.show_weightTransition()
obj.final_fit()
predicted = obj.predict(X_test, y_test)

#Comparision based on error
error_by_algo = historyGD[1]
plt.plot(np.arange(1, ITERATIONS+1), error_by_algo, c='blue')

error_by_algo = historyMomentum[1]
plt.plot(np.arange(1, ITERATIONS+1), error_by_algo, c='yellow')

error_by_algo = historyRMSP[1]
plt.plot(np.arange(1, ITERATIONS+1), error_by_algo, c='red')

error_by_algo = historyAda[1]
plt.plot(np.arange(1, ITERATIONS+1), error_by_algo, c='black')

error_by_algo = historyAdam[1]
plt.plot(np.arange(1, ITERATIONS+1), error_by_algo, c='green')

plt.legend(["GradientDescent", "Momentum", "RMSProp", "Adagrad", "Adam"])

viz = Visualize(features=features, plot_transition=True, history=historyGD)
viz.plot(type_='3d')

viz = Visualize(features=features, plot_transition=True, history=historyMomentum)
viz.plot(type_='3d')

viz = Visualize(features=features, plot_transition=True, history=historyRMSP)
viz.plot(type_='3d')

viz = Visualize(features=features, plot_transition=True, history=historyAda)
viz.plot(type_='3d')

viz = Visualize(features=features, plot_transition=True, history=historyAdam)
viz.plot(type_='3d')