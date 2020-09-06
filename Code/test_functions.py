# list of the check functions, which have to output a boolean, with True if the  test passes

#importing the libraraies
import requests
import json
import os
import keras
from training.train import *
from preprocessing.preprocess import *
from training.train import *
checkdeploymentError=""

# functions to for unit testing  

def checktrainingdataformat():
    # function that checks if the training data is in correct format 
    try:
        X_train,Y_train,X_test,Y_test = import_Xy()
        
        #Checking X_train
        if isinstance(X_train, np.ndarray) and X_train.shape==(len(X_train),28,28,1):
            print("X_train ok")
        else: print("X_train not ok")
            
        #Checking X_test   
        if isinstance(X_test, np.ndarray) and X_test.shape==(len(X_test),28,28,1):
            print("X_test ok")
        else: print("X_test not ok")
            
        #Checking y_train
        if isinstance(y_train, np.ndarray) and y_train.shape==(len(y_train),10):
            print("y_train ok")
        else: print("y_train not ok")
        
        #Checking y_test
        if isinstance(y_test, np.ndarray) and y_test.shape==(len(y_test),10):
            print("y_test ok")
        else:
            print("y_test not ok")
            
    except:
        return False
 
def checktraining():
    # function which tests whether the model is being trained properly
    try:
        #checking if the data gets imported or not
        X_train,Y_train,X_test,Y_test = import_Xy()
        X_test,Y_test = X_test[:5000],Y_test[:5000]

        #checking the training by running an epoch
        model= train_model(10,X_test,Y_test)
        return True
        
    except:
        return False
    
        
def checkmodelsaving():
    # function that tests if a model is saved properly with the required extention
    try:
        FILE_PATH = "test.pkl"
        
        #saving a model
        save_model(model,FILE_PATH)
        
        #checking the extention
        file_name, file_ext = os.path.splitext(FILE_PATH)
        if(file_ext==".pkl"):
            return True
        else: 
            return False
            
    except:
        return False
        
def checkdeployment():
    # function which tests whether the deployment is successful
    try:
       # Creating a sample data for fashion MNIST (28*28 images)
        sample_data = [0 for k in range(784)]
        headers = {'Content-Type': 'application/json'}
        r = requests.post(url = "http://127.0.0.1:5000/api" , 
                          headers = headers, json= {"data":sample_data})
        r.json()
        result = (type(r.json()[0])==int)
    except :
        result = False
        
    return result
    
def checkprecision():
    # function which tests the precision 
    try:
        # using the pre-trained model
        trained_model = keras.models.load_model('saved_model_original')
        
        # predicting the result 
        y_pred = trained_model.predict(X_test[5000:])
        
        # calculating the precision score
        from sklearn.metrics import precision_score
        precision = precision_score(y_test[5000:],y_pred)
        return precision>0.85
        
    except:
        return False
        