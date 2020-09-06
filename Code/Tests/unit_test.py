#THIS PYTHON PROGRAM IS USED FOR UNIT TESTING THE MODEL 

# importing the libraraies
import unittest
from Tests.test_functions import *
import warnings

# defining the class 
class Test_Model(unittest.TestCase):
    
    # Test to check if the input files are correctly preprocessed 
    def test_trainingdataformat(self):
        self.assertTrue(checktrainingdataformat())
        
    # Test for training to see if the model is being trained correctly 
    def test_training(self):
        self.assertTrue(checktraining())
    
    # Test to check if the model is being saved correctly as a pickle(pkl) file
    def test_modelsaving(self):
        self.assertTrue(checkmodelsaving())
      
    # Test to check if the model is being deployed correctly 
    def test_deployment(self):
        self.assertTrue(checkdeployment())
    
    # Test to check the precision of the model
    def test_precision(self):
        self.assertTrue(checkprecision())
        
     
#the following is not required if call by pytest instead of python
if __name__ == '__main__':
    unittest.main()
