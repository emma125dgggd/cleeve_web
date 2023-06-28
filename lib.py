#Lib.py
#Imports
import base64
import json
import tempfile
import sys
import os
import streamlit as st
import cv2
import numpy as np
from numpy import random
import requests
import random
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import tensorflow as tf
import matplotlib.pyplot as plt

#variables
#st.session_state.img_path= []
class_counts = {}

#functions

#file saver
def save_uploaded_file(uploadedfile):
    with open(os.path.join("Uploads",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())

@st.cache_data
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized




def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
          # Resize and pad image while meeting stride-multiple constraints
          shape = im.shape[:2]  # current shape [height, width]
          if isinstance(new_shape, int):
              new_shape = (new_shape, new_shape)

          # Scale ratio (new / old)
          r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
          if not scaleup:  # only scale down, do not scale up (for better val mAP)
              r = min(r, 1.0)

          # Compute padding
          new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
          dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

          if auto:  # minimum rectangle
              dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

          dw /= 2  # divide padding into 2 sides
          dh /= 2

          if shape[::-1] != new_unpad:  # resize
              im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
          top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
          left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
          im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
          return im, r, (dw, dh)
        
        
def count(founded_classes,im0):
  model_values=[]
  aligns=im0.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 ) 

  for i, (k, v) in enumerate(founded_classes.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    w= model_values[0]
    align_bottom=align_bottom-35                                                   
    #cv2.putText(im0, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(45,255,255),1,cv2.LINE_AA)
  return v,w

