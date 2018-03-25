import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon
from scipy.stats import randint
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

if __name__ == '__main__':
    path = 'example_data.csv'
    df = pd.read_csv(path)
    df = pd.get_dummies(df)

    # Create a 3d scatter plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['x'], df['y'], df['z'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    # Create a 2d scatter plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(df['x'], df['z'])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    plt.show()

    # We will train the model to predict Z and the category of the data from the X and Y values 

    # Create empty Dataframes
    X = pd.DataFrame()
    Y = pd.DataFrame()

    # Add data input data to the X Dataframe
    X['x'] = df['x']
    X['y'] = df['y']

    # Add label data to the Y Dataframe
    # Because we one-hot encoded, loop over column names and add the columns not used in the X Dataframe
    for col in df.columns.values:
        if col not in ['x', 'y']:
            Y[col] = df[col]

    # Create separate datasets for training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # Create pipelines for training machine learning models
    pipe_clf = Pipeline([('scl', StandardScaler()), ('poly', PolynomialFeatures()), ('clf', LogisticRegression())])
    pipe_regr = Pipeline([('scl', StandardScaler()), ('poly', PolynomialFeatures()), ('linear', LinearRegression(fit_intercept=False))])

    # Create the randomized searches
    c_range_clf = expon(scale=1)
    tol_range = expon(scale=1e-4)
    degree_range = randint(1, 10)

    param_dist_clf = {'poly__degree': degree_range, 'clf__C': c_range_clf, 'clf__tol': tol_range, 'clf__solver': ['newton-cg', 'lbfgs', 'liblinear']}
    param_dist_regr = {'poly__degree': degree_range}

    rs_clf = RandomizedSearchCV(pipe_clf, param_dist_clf, n_iter=300, scoring='accuracy', cv=10, n_jobs=-1, verbose=1)
    rs_regr = RandomizedSearchCV(pipe_regr, param_dist_regr, n_iter=50, scoring='r2', cv=10, n_jobs=-1, verbose=1)

    # Split the test and training labels for classification and regression
    y_train_clf = pd.DataFrame()
    y_test_clf = pd.DataFrame()

    y_train_regr = pd.DataFrame()
    y_test_regr = pd.DataFrame()

    for col in y_train.columns.values:
        if 'z' in col:
            y_train_regr[col] = y_train[col]
            y_test_regr[col] = y_test[col]
        else:
            y_train_clf[col] = y_train[col]
            y_test_clf[col] = y_test[col]

    mo_clf = MultiOutputClassifier(rs_clf)

    # Fit the data to the models
    print('Fitting data')
    mo_clf.fit(x_train, y_train_clf)
    rs_regr.fit(x_train, y_train_regr)

    # Print the results of the fit on the test data
    print('Test classification score: %.3f' % mo_clf.score(x_test, y_test_clf))
    print('Test regression R2 score: %.3f' % rs_regr.score(x_test, y_test_regr))

    # Plot the decision surfaces of the classifier and regressor
    x = pd.DataFrame(np.linspace(0, 5, 25))
    y = pd.DataFrame(np.linspace(0, 5, 25))

    # Create a grid to plot our predicted values over
    surf_x = pd.DataFrame(np.array(np.meshgrid(x, y, )).T.reshape(-1, 2))
    surf_z = pd.DataFrame()    

    # Predict a value for each (x, y) pair in the grid
    surf_z['z'] = [x[0] for x in rs_regr.predict(surf_x).tolist()]
    
    # Translate our one-hot encoded labels back into a single list of string values
    category_list = []
    for x in mo_clf.predict(surf_x).tolist():
        if x[0] == 0 and x[1] == 1:
            category_list += ['pos']
        elif x[0] == 1 and x[1] == 0:
            category_list += ['neg']
        else:
            category_list += ['unk']

    surf_z['cat'] = category_list
    
    # Separate our values into separate surfaces so we can easily color the different categories from the classifier
    pos_surf = surf_z.copy()
    neg_surf = surf_z.copy()
    unk_surf = surf_z.copy()
    
    # Fill in values that don't match the label with NaN
    pos_surf[pos_surf['cat'] != 'pos'] = np.nan
    neg_surf[neg_surf['cat'] != 'neg'] = np.nan
    unk_surf[unk_surf['cat'] != 'unk'] = np.nan

    # Make the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(surf_x[0], surf_x[1], pos_surf['z'], color='b')
    ax.scatter(surf_x[0], surf_x[1], neg_surf['z'], color='r')
    ax.scatter(surf_x[0], surf_x[1], unk_surf['z'], color='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    