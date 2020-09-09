#THIS PYTHON PROGRAM IS USED FOR TRAINING THE MACHINE LEARNING MODEL AND SAVING IT

# defining the path for pre processing the 
preprocess_path = "Code/preprocessing"

# importing modules
import sys
import joblib
import logging

# importing the preprocessing python file
sys.path.append(preprocess_path)
from preprocessing.preprocess import import_Xy

# importing the deep learning modules
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten

# function to train the model
def train_model(epoch,X_train,Y_train):

    # defing the model using the sequential class
    classifier = Sequential()
    classifier.load_weights("saved_model_original_1/variables/variables.index")
    # adding layers in the model
    # first conv and maxpool layer
    classifier.add(Convolution2D(64, [5,5], input_shape = (28, 28, 1), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = [2, 2]))

    # second conv and maxpool layer
    #classifier.add(Convolution2D(64, [2,2], activation = 'relu'))
    #classifier.add(MaxPooling2D(pool_size = [2, 2]))

    # third conv and maxpool layer
    classifier.add(Convolution2D(64 , [5,5], activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = [2, 2]))

    # flattening layer
    classifier.add(Flatten())

    # first dense layer with input and a dropout after that
    classifier.add(Dense(64, activation = 'relu'))
    
    # third dense layer
    classifier.add(Dense(32, activation = 'relu'))
    
    # fourth dense layer for output
    classifier.add(Dense(10, activation = 'softmax'))

    # compiling the neural network
    classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(classifier.summary())
    
    # fitting the data into the model
    classifier.fit(X_train, Y_train, batch_size = 500, 
                   epochs = epoch, validation_split = 0.2)
    
    # saving the model
    classifier.save('saved_model_original_1')

# function to save the model as a pkl file for deployment
#def save_model(model,model_path):
    #joblib.dump(model,model_path)

# main function
if __name__ == "__main__":
    # importing the data
    X_train,Y_train,X_test,Y_test = import_Xy()
    
    # Training the model
    model= train_model(10,X_train,Y_train)
    
    # saving the model
    #save_model(model,"Code.pkl")
 