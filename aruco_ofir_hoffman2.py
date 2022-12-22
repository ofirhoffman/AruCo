import streamlit as st
from skimage import io, filters, feature, img_as_float, img_as_ubyte,measure,util,color
from skimage.color import label2rgb, rgb2gray
import skimage
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
# vars
DEMO_IMAGE = 'demo.jpg' # a demo image for the segmentation page, if none is uploaded
favicon = 'favicon.png'

# main page
st.set_page_config(page_title='AruCo - ofir hoffman', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title('calculating area of leaf using AruCo , by Ofir Hoffman')

# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)

st.sidebar.title('Segmentation Sidebar')
st.sidebar.subheader('Site Pages')
