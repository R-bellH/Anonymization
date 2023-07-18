import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def linear_regression(X, y):
    __name__='linear regression'
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    # linear regression model
    model = LinearRegression()
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    return y_test, y_pred

def random_forest(X, y):
    __name__='random forest'
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # random forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

def heart(k_list, model):
    print('heart')
    mses=[]
    for k in k_list:
        path=rf"data\heart-disease-{k}-anonymized.csv"
        data = pd.read_csv(path)
        y = data["target"]
        X = data.drop("target", axis=1)
        X = pd.get_dummies(X)
        y_test, y_pred = model(X, y)
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
        print(f"mse for {k} k is {mse}")
    return mses
def adult(k_list, model):
    print("adult")
    mses = []
    for k in k_list:
        path = rf"data\example-{k}-anonymized.csv"
        data = pd.read_csv(path,delimiter=';')
        data['salary-class']=data['salary-class'].map({'<=50K':0,'>50K':1})
        y = data["salary-class"]
        data = data.drop("salary-class", axis=1)
        X = pd.get_dummies(data)
        y_test, y_pred = model(X, y)
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
        print(f"mse for {k} k is {mse}")
    return mses

k_list = [1, 2, 3, 5, 7, 10, 25, 50]
for model in [linear_regression,random_forest]:
    print("**********"+model.__name__+"**********")
    mses_heart=heart(k_list, model)
    mses_adult=adult(k_list, model)
    # Plot the results
    plt.plot(k_list, mses_heart, label="heart")
    plt.plot(k_list, mses_adult, label="adult")
    plt.title(f"{model.__name__} mse for different k")
    plt.xlabel("k")
    plt.ylabel("mse")
    plt.xticks(k_list)
    plt.gca().yaxis.set_major_formatter(lambda x, _: '{:.0%}'.format(x))
    plt.legend()
    plt.show()