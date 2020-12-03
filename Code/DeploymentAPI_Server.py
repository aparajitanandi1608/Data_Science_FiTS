#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
from flask import Flask, jsonify
from flask import abort
from flask import request
import numpy as np
import json
import os
import pickle
import fastai
from fastai.vision import learner
from fastai.vision import models
import torch
from PIL import Image
from fastai.vision import *
from fastai.vision.all import *
import io                   
import base64                  
import logging             


# In[2]:


Pkl_Filename = "resnet18_pkl.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)


# In[3]:


#print(Pickled_Model.summary())


# In[ ]:


app = Flask(__name__)

print("launching the server....")

@app.route('/api', methods=["POST","GET"])
def ModelEval(verbose = True):
    
    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    x = "The shape of the given image is:" + str(img_arr.shape)
    
    pred = Pickled_Model.predict(img_arr)
    p=str(pred[0])
    pred_str = "The predicted class of the image is : " + str(p)
    
    return pred_str
    
if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '4242'))
    except ValueError:
        PORT = 4242
    app.run(port = PORT, host = '0.0.0.0')

