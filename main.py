#Cleeve AI

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
from lib import *

#variables
#paths = []
def counter(json):
    with open(json, "r") as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    
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



#Streamlit UI
st.title('Cleeve AI')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Define icon menu options
icon_options = {
    
    "Home": '<i class="fas fa-home"></i>',
    
    
}

# Display icon menu
selected_icon = st.sidebar.selectbox("Menu", list(icon_options.values()), format_func=lambda x: list(icon_options.keys())[list(icon_options.values()).index(x)])

# Handle login and signup options
if selected_icon == icon_options["Home"]:
    st.markdown('Cleeve AI is an Assistive Medical Diagnostic Software that speeds up the turn around time of Lab Diagnosis ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    app_mode = st.sidebar.selectbox('Diagnosis',
                                    ["Demo", "Slide Image", 'Slide Video']
                                    )
    
    if app_mode == 'Slide Image':

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 320px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 320px;
                margin-left: -400px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'],
                                                   accept_multiple_files=True)
        

        if img_file_buffer is not None:
            for UploadedFile in img_file_buffer:
                print(UploadedFile)
                file_details = {"filename": UploadedFile.name, "file_type": UploadedFile.type}
                save_uploaded_file(UploadedFile)
                uploaded_image = "Uploads/{}".format(UploadedFile.name)
                with open(uploaded_image, "rb") as file:
                    image_data = file.read()
                    # Encode the image data as base64
                encoded_image_data = base64.b64encode(image_data).decode("utf-8")
                
                # Create a dictionary for the JSON object
                json_data = {
                    "filename": UploadedFile.name,
                    "image_data": encoded_image_data
                }
                
                # Convert the dictionary to JSON
                json_string = json.dumps(json_data)
                json_file_path = "json.json"  # Replace with the desired file path
                with open(json_file_path, "w") as json_file:
                    json.dump(json_data, json_file)
                st.image(uploaded_image)
                
                image, count = counter(json_file_path)
                data = json.loads(image)

                # Retrieve the image data from the JSON
                image_data = data["image_data"]
                
                # Decode the base64-encoded image data
                decoded_image_data = base64.b64decode(image_data)
                
                # Save the image to a file
                with open(data["filename"], "wb") as file:
                    file.write(decoded_image_data)
                
                st.image(data["filename"])
                
                count_data = json.loads(count)

                # Access the decoded data
                Parasite = count_data["Parasite"]
                WBC = count_data["WBC"]
                
                # Print or use the decoded data as needed
                st.success(f'Parasite Detected: {Parasite} ')
                st.success(f'WBC Detected: {WBC} ')
                st.write(image)
                st.write(count)
                
                
    
    elif app_mode =='Demo':
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )
        st.video('result_compressed.mp4')

    elif app_mode == 'Slide Video':
        st.video('result_compressed.mp4')

        

                
                


                
               




