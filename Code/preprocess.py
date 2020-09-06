#THIS PYTHON PROGRAM IS USED FOR LOADING AND PREPROCESSING OF THE DATA 

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import keras


def import_Xy(path_train="fashion-mnist_train.csv",
              path_test="fashion-mnist_test.csv",label_name="label"):
    # importng the data from the paths which are there by default
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    
    # seperating the data into training and test sets
    X_train,y_train = df_train.drop("label",axis=1),df_train["label"]
    X_test,y_test = df_test.drop("label",axis=1),df_test["label"]

    # reshaping the dataset
    X_train = np.array(X_train).reshape(-1,28,28,1)
    X_test = np.array(X_test).reshape(-1,28,28,1)
    print(X_train.shape)
    print(X_test.shape)

    # Normalization
    X_train=X_train/255
    X_test=X_test/255

    # converting the labels into encodings
    from sklearn.preprocessing import LabelBinarizer
    lb=LabelBinarizer()
    y_train=lb.fit_transform(y_train)
    y_test=lb.fit_transform(y_test)

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    return X_train,y_train, X_test,y_test

# Main function
if __name__=="__main__":
    # importing the data
    X_train, y_train, X_test, y_test = import_Xy()
    print("imported data")
    