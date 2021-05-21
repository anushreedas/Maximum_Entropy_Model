import numpy as np
import matplotlib.pyplot as plt

"""
q1b.py

This program train the previous maximum entropy model on the data and then plot its decision boundary.

@author: Anushree Sitaram Das (ad1707)
"""


def softmax(X):
    """
    Activation function for regression model.
    It takes as input a vector z of K real numbers,
    and normalizes it into a probability distribution consisting of
    K probabilities proportional to the exponentials of the input numbers.
    :param X:   input array
    :return:
    """
    return (np.exp(X.T) / np.sum(np.exp(X), axis=1)).T


def loss_function(W, b, X, y,lamb=1):
    """
    Calculates loss using cost function and
    calculates the cost function's partial derivative

    :param W:       Weights
    :param b:       Bias
    :param X:       Input Features
    :param y:       Output Class
    :param lamb:    lambda value
    :return: Loss, partial gradient descents for Weights and Bias
    """

    N = X.shape[0]

    p = softmax(X.dot(W) + b)

    # cost function
    loss = (-1 / N) * np.sum(y * np.log(p)) + (lamb/2)*np.sum(W*W)

    # partial gradient descent of cost function for Weights
    gradW = (1 / N) * (np.dot(X.T, (p - y))) + (lamb * W)
    # partial gradient descent of cost function for Bias
    gradb = (1 / N) * (np.sum((p - y), axis=0))

    return loss, gradW, gradb


def vertor_to_matrix(y):
    """
    Converts output class vector to matrix.
    Ex:
    vector = [0,1,2,1]
    matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]]
    :param y:   vector
    :return:    matrix
    """
    return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)


def get_parameters(X,Y,epochs):
    """
    Calculates optimal values for weights and bias via regression

    :param X:       Input features
    :param Y:       Output class
    :param epochs:  Number of epochs
    :return:        Optimal weights and bias
    """
    # initialize Weights
    W = np.random.randn(X.shape[1], len(np.unique(Y)))
    # initialize Bias
    b = np.zeros(len(np.unique(Y)))
    # convert output vector to matrix
    y = vertor_to_matrix(Y)

    # stores loss for each epoch
    losses = []
    # learning rate
    learningRate = 0.01

    for i in range(0, epochs):
        # get loss and gradient descents for weights and bias
        loss, gradW, gradb = loss_function(W, b, X, y)
        losses.append(loss)
        # update weights and bias
        W = W - (learningRate * gradW)
        b = b - (learningRate * gradb)

    # Plot Losses
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig('losses_q1b.png')
    plt.show()

    return W,b


def predict(X, W, b):
    """
    Predict class for given vector of features

    :param X: Input features
    :param W: Weights
    :param b: Bias
    :return:  Class of given input features
    """
    prob = softmax(X.dot(W) + b)
    # return index of the element with maximum value
    return prob.argmax(axis=1)


def get_accuracy(X,y,W,b):
    """
    Calculates accuracy of predictions for the given model
    :param X: Input features
    :param y: Output class
    :param W: Weights
    :param b: Bias
    :return:  Accuracy in percentage
    """
    pred = predict(X,W,b)
    sum = 0
    for i in range(len(pred)):
        if pred[i]==y[i]:
            sum+=1
    accuracy = sum/(float(len(y)))
    return accuracy*100


def gradient_checking(X, Y, W, b):
    """
    Performs gradient-checking.

    :param X: Input features
    :param y: Output class
    :param W: Weights
    :param b: Bias
    :return:  None
    """
    # convert output vector to matrix
    y = vertor_to_matrix(Y)
    epsilon = 10e-4

    # check approximations of derivatives for given weights
    print("Checking gradients of Weights:")
    for i in range(len(W)):
        for j in range(len(W[0])):
            Wnew = W.copy()

            Wnew[i][j] = W[i][j] + epsilon
            Jplus, gradW, gradb = loss_function(Wnew, b, X, y)

            Wnew[i][j] =  W[i][j] - epsilon
            Jminus, gradW, gradb = loss_function(Wnew, b, X, y)

            # approximation of derivative
            if (Jplus - Jminus) / (2 * epsilon) < 1e-3:
                print("CORRECT")
            else:
                print("INCORRECT", (Jplus - Jminus) / (2 * epsilon))

    # check approximations of derivatives for given weights
    print("Checking gradients of Bias:")
    for i in range(len(b)):
        bnew = b.copy()

        bnew[i] = b[i] + epsilon
        Jplus, gradW, gradb = loss_function(Wnew, b, X, y)

        bnew[i] =  b[i] - epsilon
        Jminus, gradW, gradb = loss_function(W, bnew, X, y)

        # approximation of derivative
        if (Jplus - Jminus) / (2 * epsilon) < 1e-3:
            print("CORRECT")
        else:
            print("INCORRECT", (Jplus - Jminus) / (2 * epsilon))


def plot_decision_boundary(X,y, W, b):
    """
    Plots decision boundary for the given dataset using the optimal parameters given
    :param X: Input features
    :param W: Weights
    :param b: Bias
    :return:  None
    """

    plt.grid()
    plt.xlabel('feature 1', size=20)
    plt.ylabel('feature 2', size=20)

    # get both feature values for plotting in 2d place
    feature1 = X[:, 0]
    feature2 = X[:, 1]

    # assign color for each input according to its output class
    colors = []
    for c in y:
        if c == 0:
            colors.append('green')
        else:
            if c == 1:
                colors.append('red')
            else:
                colors.append('blue')

    # plot features
    plt.scatter(feature1, feature2, s=5, color=colors)

    # build heatmap
    # find min and max values of both features
    x_min, x_max = feature1.min() - 1, feature1.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1

    # Predict class for all combinations of values of both features
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    Xnew = np.c_[xx.ravel(), yy.ravel()]
    pred = predict(Xnew,W,b)
    pred = pred.reshape(xx.shape)

    plt.contourf(xx,yy,pred,alpha=0.5)

    plt.savefig('decision_boundary_q1b.png')
    plt.show()


if __name__ == "__main__":
    # load data
    data = np.genfromtxt('spiral_train.dat', delimiter=',')
    # array of features
    X = np.array(data[:,:-1])
    # array of output class for corresponding feature set
    y = np.array(data[:, -1])
    # number of epochs
    epochs = 1000

    # get optimal parameters
    (W, b) = get_parameters(X, y, epochs)

    # perform gradient checking
    gradient_checking(X, y, W, b)

    # print optimal parameters
    print("Optimal Parameters(weights and bias) are: \n", W, b, "\n")

    # calculate accuracy
    print("Accuracy:", get_accuracy(X, y, W, b), "%")

    plot_decision_boundary(X, y, W, b)


