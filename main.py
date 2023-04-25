#Cleeve AI

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
from lib import *




#functions

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
        

        if img_file_buffer is None:
            img_file = st.file_uploader("Upload an image", type=["jpg"],
                                        accept_multiple_files=True, key=1)
            


                
               


else:
    # Do something for other menu options
    pass



