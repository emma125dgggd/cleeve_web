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
        

        if img_file_buffer is not None:
            for UploadedFile in img_file_buffer:
                print(UploadedFile)
                file_details = {"filename": UploadedFile.name, "file_type": UploadedFile.type}
                save_uploaded_file(UploadedFile)
                paths = []
                uploaded_image = "Uploads/{}".format(UploadedFile.name)
                paths = paths.append(uploaded_image)
                st.sidebar.text('Original Image')
                st.sidebar.image(uploaded_image)
                process_slide(uploaded_image)


                
               


else:
    # Do something for other menu options
    pass



