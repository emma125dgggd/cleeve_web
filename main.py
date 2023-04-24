#Cleeve AI

#Imports
import base64
import tempfile
import sys
import os
import streamlit as st
import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel





#functions
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)





#file saver
def save_uploaded_file(uploadedfile):
    with open(os.path.join(uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())


 




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




sign = None
def signup():
    st.header("Create an account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm password", type="password")

    

def login():
    st.header("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    


# Define icon menu options
icon_options = {
    "Login": '<img src="https://img.icons8.com/plumpy/24/000000/enter-2.png>',
    "Signup": '<i class="fas fa-user-plus"></i>',
    "Home": '<i class="fas fa-home"></i>',
    "Profile": '<i class="fas fa-user"></i>',
    "About Us": '<i class="fas fa-comment"></i>',
    "Contact": '<i class="fas fa-envelope"></i>',
    "Settings": '<i class="fas fa-cog"></i>',

}

# Display icon menu
selected_icon = st.sidebar.selectbox("Menu", list(icon_options.values()), format_func=lambda x: list(icon_options.keys())[list(icon_options.values()).index(x)])


def home():
    st.markdown(
        'Cleeve AI is an Assistive Medical Diagnostic Software that speeds up the turn around time of Lab Diagnosis ')
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
    if app_mode =='Demo':
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

    if app_mode == 'Slide Video':
        st.video('result_compressed.mp4')

        

    elif app_mode == 'Slide Image':

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
        

        if img_file_buffer is None:'
            img_file = st.file_uploader("Upload an image", type=["jpg"],
                                        accept_multiple_files=True, key=1)
            def counter():
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
                pred = model(img, augment= False)[0]


                pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
                t2 = time_synchronized()
                for i, det in enumerate(pred):
                  s = ''
                  s += '%gx%g ' % img.shape[2:]  # print string
                  gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                  if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for c in det[:, -1].unique():
                      n = (det[:, -1] == c).sum()  # detections per class
                      s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                   
            counter()
                
               

if selected_icon == icon_options["Signup"]:
    signup()


elif selected_icon == icon_options["Login"]:
    login()
else:
    # Do something for other menu options
    pass



