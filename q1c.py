import numpy as np
import matplotlib.pyplot as plt
"""
q1c.py

This program fits the previous maximum entropy model to the given IRIS dataset,
where parameter optimization operates with mini-batches and
estimates accuracy by using the validation/development dataset given.

@author: Anushree Sitaram Das (ad1707)
"""


def create_mini_batch(X,y,batchSize):
    """
    Divides given dataset into mini batches

    :param X:           Input Features
    :param y:           Output Class
    :param batchSize:   size of each batch
    :return:            mini batches
    """
    # stores the mini batches
    mini_batches = []

    data = np.column_stack((X, y))
    np.random.shuffle(data)

    # total number of batches
    n_minibatches = data.shape[0] // batchSize

    # divide dataset into small batches of equal sizes
    for i in range(n_minibatches):
        mini_batch = data[i * batchSize:(i + 1) * batchSize, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, ))
        mini_batches.append((X_mini, Y_mini))
    # last batch of leftover data
    if data.shape[0] % batchSize != 0:
        mini_batch = data[n_minibatches * batchSize:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, ))
        mini_batches.append((X_mini, Y_mini))

    return mini_batches


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


def vertor_to_matrix(y,c):
    """
    Converts output class vector to matrix.
    Ex:
    vector = [0,1,2,1]
    matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]]
    :param y:   vector
    :param c:   total number of classes
    :return:    matrix
    """
    return (np.arange(c) == y[:, None]).astype(float)


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


def get_parameters(X_train, y_train,X_test,y_test,epochs,batchSize):
    # total number of classes
    c = len(np.unique(y_train))
    # initialize Weights
    W = np.random.randn(X_train.shape[1], c)
    # initialize Bias
    b = np.zeros(c)

    # stores loss for each epoch for training
    train_loss = []
    # stores accuracy for each epoch for training
    train_accuracy =[]
    # stores loss for each epoch for validation
    test_loss = []
    # stores accuracy for each epoch for validation
    test_accuracy = []

    # learning rate
    learningRate = 0.01

    for i in range(0, epochs):
        # get mini batches of given training dataset
        mini_batches = create_mini_batch(X_train, y_train, batchSize)

        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            # convert output vector to matrix
            y_mini = vertor_to_matrix(Y_mini,c)

            # get gradient descents for weights and bias
            loss, gradW, gradb = loss_function(W,b,X_mini, y_mini)

            # update weights and bias
            W = W - (learningRate * gradW)
            b = b - (learningRate * gradb)

        # convert output vector to matrix
        Y_train = vertor_to_matrix(y_train,c)
        # get loss for current weights and bias for training set
        loss, gradW, gradb = loss_function(W, b, X_train, Y_train)
        train_loss.append(loss)
        # get accuracy for current weights and bias for training set
        accuracy = get_accuracy(X_train, y_train, W, b)
        train_accuracy.append(accuracy)

        # convert output vector to matrix
        Y_test = vertor_to_matrix(y_test,c)
        # get loss for current weights and bias for validation set
        loss, gradW, gradb = loss_function(W, b, X_test,Y_test)
        test_loss.append(loss)
        # get accuracy for current weights and bias for validation set
        accuracy = get_accuracy(X_test,y_test, W, b)
        test_accuracy.append(accuracy)

    # Plot losses for training and validation datasets
    plt.plot(train_loss, '--',alpha = 1.0)
    plt.plot(test_loss, alpha = 0.5)
    plt.savefig('losses_q1c.png')
    plt.show()

    # Plot accuracy for training and validation datasets
    plt.plot(train_accuracy, '--',alpha = 1.0)
    plt.plot(test_accuracy, alpha = 0.5)
    plt.savefig('accuracy_q1c.png')
    plt.show()

    return W,b



def gradient_checking(X, Y, W, b):
    """
    Performs gradient-checking.

    :param X: Input features
    :param y: Output class
    :param W: Weights
    :param b: Bias
    :return:  None
    """
    # total number of classes
    c = len(np.unique(y_train))
    # convert output vector to matrix
    y = vertor_to_matrix(Y,c)
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


if __name__ == "__main__":
    # load training data
    data_train = np.genfromtxt('iris_train.dat', delimiter=',')
    # array of features
    X_train = np.array(data_train[:,:-1])
    # array of output class for corresponding feature set
    y_train = np.array(data_train[:,-1])

    # load validation data
    data_test = np.genfromtxt('iris_test.dat', delimiter=',')
    # array of features
    X_test = np.array(data_test[:, :-1])
    # array of output class for corresponding feature set
    y_test = np.array(data_test[:, -1])

    # number of epochs
    epochs = 10000
    # mini batch size
    batchSize = 25

    # get optimal parameters
    (W,b) = get_parameters(X_train, y_train,X_test,y_test,epochs,batchSize)

    # perform gradient checking
    gradient_checking(X_train, y_train, W, b)

    # print optimal parameters
    print("Optimal Parameters are: \n", W,b, "\n")

    # calculate accuracy
    print("Accuracy:",get_accuracy(X_test,y_test,W,b),"%")

