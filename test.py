import hyperopt.pyll
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


@scope.define
def foo(a, b=0):
     print('runing foo', a, b)
     return a + b / 2

if __name__ == "__main__":
    # # -- this will print 0, foo is called as usual.
    # print(foo(0))
    #
    # # In describing search spaces you can use `foo` as you
    # # would in normal Python. These two calls will not actually call foo,
    # # they just record that foo should be called to evaluate the graph.
    #
    # space1 = scope.foo(hp.uniform('a', 0, 10))
    # space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1))
    #
    # # -- this will print an pyll.Apply node
    # print(space1)
    #
    # # -- this will draw a sample by running foo()
    # print(hyperopt.pyll.stochastic.sample(space1))


    # linear model test
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # The coefficients
    params = regr.get_params()
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
             linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

