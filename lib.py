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

def counter(json):
    with open(json, "r") as json_file:
        # Load the JSON data from the file
        json_data = json.load(json_file)
    
    #data = json.loads(json)

    # Retrieve the image data from the JSON
    image_data = data["image_data"]
    image = data["filename"]
    # Decode the base64-encoded image data
    decoded_image_data = base64.b64decode(image_data)
    
    # Save the image to a file
    with open(image, "wb") as file:
        file.write(decoded_image_data)
    
    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'

        model = attempt_load(weights, map_location='cpu') 
        #model = torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu'))

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        img0 = cv2.imread(image)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(opt['classes'].index(class_name))

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes,
                                   agnostic=False)
        t2 = time_synchronized()
        v=0
        w=0

        sum =[]
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  
                founded_classes={} # Creating a dict to storage our detected items
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    class_index=int(c)
                    count_of_object=int(n)
                    founded_classes[names[class_index]]=int(n)
                    v, w =count(founded_classes=founded_classes,im0=img0)  # Applying counter function
                    #total(founded_classes,im0=img0,total_last) 

                crp_cnt = 0

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=None, color=colors[int(cls)], line_thickness=3)

                    # crop
                    # crop an image based on coordinates
                    object_coordinates = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    cropobj = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                    # save crop part
                    crop_file_path = os.path.join("crop", str(uuid.uuid4()) + ".jpg")
                    cv2.imwrite(crop_file_path, cropobj)
                    crp_cnt = crp_cnt + 1

        img_path = os.path.join("runs", data["filename"])
        
        st.session_state.img_path = img_path
        print(img_path)
        cv2.imwrite(img_path, img0)
        with open(img_path, "rb") as file:
            image_data = file.read()

        # Encode the image data as base64
        encoded_image_data = base64.b64encode(image_data).decode("utf-8")
        
        # Create a dictionary for the JSON object
        json_image = {
            "filename": data["filename"],
            "image_data": encoded_image_data
            
        }
        
        # Convert the dictionary to JSON
        json_string = json.dumps(json_image)
        count_dict = {
            "Parasite": v,
            "WBC": w
        }
        
        # Convert the dictionary to JSON
        json_count = json.dumps(count_dict)
        
        # Print or use the JSON string as needed
        

    return json_string, json_count

