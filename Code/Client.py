#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2

image_path = "./yellow-shirt.png"
image = cv2.imread(image_path)
plt.imshow(image)


# In[2]:


import requests
import json
import base64

image_file = "yellow-shirt.png"


# In[3]:


with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
  
payload = json.dumps({"image": im_b64, "other_key": "value"})
response = requests.post(url = "http://13.81.12.56:4242/api", data=payload, headers=headers)


# In[4]:


response.content

