#Lib.py
#Imports
import base64
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
        
        
def process_slide(uploaded_image, UploadedFile):
          # Load the TFLite model and allocate tensors.
          interpreter = tf.lite.Interpreter(model_path = "yolov7_model.tflite")


          #Name of the classes according to class indices.
          names = ['Pf']

          #Creating random colors for bounding box visualization.
          colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

          #Load and preprocess the image.
          img = cv2.imread(uploaded_image)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

          image = img.copy()
          image, ratio, dwdh = letterbox(image, auto=False)
          image = image.transpose((2, 0, 1))
          image = np.expand_dims(image, 0)
          image = np.ascontiguousarray(image)

          im = image.astype(np.float32)
          im /= 255

         

          #Allocate tensors.
          interpreter.allocate_tensors()
          # Get input and output tensors.
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()

          # Test the model on random input data.
          input_shape = input_details[0]['shape']
          interpreter.set_tensor(input_details[0]['index'], im)

          interpreter.invoke()

          # The function `get_tensor()` returns a copy of the tensor data.
          # Use `tensor()` in order to get a pointer to the tensor.
          output_data = interpreter.get_tensor(output_details[0]['index'])
          #Allocate tensors.
          interpreter.allocate_tensors()
          # Get input and output tensors.
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()

          # Test the model on random input data.
          input_shape = input_details[0]['shape']
          interpreter.set_tensor(input_details[0]['index'], im)

          interpreter.invoke()

          # The function `get_tensor()` returns a copy of the tensor data.
          # Use `tensor()` in order to get a pointer to the tensor.
          output_data = interpreter.get_tensor(output_details[0]['index'])
          def set_input_tensor(interpreter, image):
            """Sets the input tensor."""
            tensor_index = interpreter.get_input_details()[0]['index']
            input_tensor = interpreter.tensor(tensor_index)()[0]
            input_tensor[:, :] = image

 
          def get_output_tensor(interpreter, index):
            """Returns the output tensor at the given index."""
            output_details = interpreter.get_output_details()[index]
            tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
            return tensor


          def detect_objects(interpreter, image, threshold):
            """Returns a list of detection results, each a dictionary of object info."""
            set_input_tensor(interpreter, image)
            interpreter.invoke()

            # Get all output details
            boxes = get_output_tensor(interpreter, 0)
            classes = get_output_tensor(interpreter, 1)
            scores = get_output_tensor(interpreter, 2)
            count = int(get_output_tensor(interpreter, 3))

            results = []
            for i in range(count):
                if scores[i] >= threshold:
                    result = {
                        'bounding_box': boxes[i],
                        'class_id': classes[i],
                        'score': scores[i]
                    }
                    results.append(result)
            return results

          results = detect_objects(interpreter, image_pred,threshold =0.2)



          ori_images = [img.copy()]
          class_counts = {}
          for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
              image = ori_images[int(batch_id)]
              box = np.array([x0,y0,x1,y1])
              box -= np.array(dwdh*2)
              box /= ratio
              box = box.round().astype(np.int32).tolist()
              cls_id = int(cls_id)
              score = round(float(score),3)
              name = names[cls_id]
              if name not in class_counts:
                class_counts[name] = 1
              else:
                class_counts[name] += 1
              color = colors[name]
              name += ' '+str(score)
              cv2.rectangle(image,box[:2],box[2:],color,2)
              cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
              
                
          
                
          img_path = os.path.join("Detected_Images", UploadedFile.name)
          print(img_path)        
          cv2.imwrite(img_path, ori_images[0]) 
          #if "img_path" not in st.session_state:
              #st.session_state.img_path = []
          #    st.session_state.img_path= st.session_state.img_path.append(img_path)
          st.image(ori_images[0], use_column_width=True, channels="RGB")
          st.write(img_path)
          #st.write(st.session_state.img_path)
          st.write(class_counts)
          return img_path

        
