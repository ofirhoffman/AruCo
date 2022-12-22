import streamlit as st
from skimage import io, filters, feature, img_as_float, img_as_ubyte,measure,util,color
from skimage.color import label2rgb, rgb2gray
import skimage
import matplotlib.pyplot as plt
import cv2
import numpy as np
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

# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)
@st.cache()

# take an image, and return a resized that fits our page
def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized

# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'Segment an Image'])

# About page
if app_mode == 'About App':
    st.markdown('In this app we will calculating arae of leaf using AruCo')
    
    
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

    # add a video to the page
    st.image(DEMO_IMAGE , caption= 'an example of how the image should be')


    st.markdown('''
                ## About the app \n
                Hey, this web app is a great one to calculate arae of leaf using AruCo. \n
                your image should have only: your leaf, AruCo marker, white background. \n
                Enjoy! Ofir


                ''') 
# Run image
# Run image
if app_mode == 'Segment an Image':
    
    st.sidebar.markdown('---') # adds a devider (a line)
    
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

        # read an image from the user
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    # assign the uplodaed image from the buffer, by reading it in
    if img_file_buffer is not None:
        img = io.imread(img_file_buffer)
    else: # if no image was uploaded, then segment the demo image
        demo_image = DEMO_IMAGE
        img = io.imread(demo_image)

    # display on the sidebar the uploaded image
    st.sidebar.text('Original Image')
    st.sidebar.image(img)

    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        # Draw polygon around the marker
    int_corners = np.int0(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 0)

        # Aruco Area
    aruco_area = cv2.contourArea (corners[0])

    cv2.fillPoly(img, pts =[int_corners[0][0]], color=(0,0,0))
        # Pixel to cm ratio
    pixel_cm_ratio = 5*5 / aruco_area# since the AruCo is 5*5 cm, so we devide 25 cm*cm by the number of pixels

    plt.imshow(img)

    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case, MxN

    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    pixel_values = np.float32(pixel_values)

    img.reshape((-1,3)).shape # 602 * 900 = 541800

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 3

    attempts = 10
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    # show the image
    #plt.imshow(segmented_image)
    #plt.show()

    #print(centers)

    lst = list(np.median(centers,axis = 0))

    result_1 = [int(item) for item in lst]

    for i,center in enumerate(centers):
     if np.all(center == (result_1)):
        grass_center_index = i
    

    # copy source img
    masked_image = img.copy()

    # convert to the shape of a vector of pixel values (like suits for kmeans)
    masked_image = masked_image.reshape((-1, 3))

    list_of_cluster_numbers_to_exclude = list(range(k)) # create a list that has the number from 0 to k-1
    list_of_cluster_numbers_to_exclude.remove(grass_center_index) # remove the cluster of grass that we want to keep, and not black out
    for cluster in list_of_cluster_numbers_to_exclude:
     masked_image[labels== cluster] = [0, 0, 0] # black all clusters except cluster 3

    # convert back to original shape
    masked_image = masked_image.reshape(img.shape)

    # show the image
    plt.imshow(masked_image)

    masked_image_grayscale = rgb2gray(masked_image)

    # count how many pixels are in the foreground and bg
    leaf_count = np.sum(np.array(masked_image_grayscale) >0)
    bg_count = np.sum(np.array(masked_image_grayscale) ==0)

    #print('Leaf px count:', leaf_count, 'px')
    #print('Area:', leaf_count*pixel_cm_ratio, 'cm\N{SUPERSCRIPT TWO},', 'which is:',  f'{0.0001*leaf_count*pixel_cm_ratio:.3f}', 'm\N{SUPERSCRIPT TWO}')


    
    
    # Display the result on the right (main frame)
    st.subheader('segmanted image')
    st.image(masked_image, use_column_width=True)
    st.write('Area:', leaf_count*pixel_cm_ratio, 'cm\N{SUPERSCRIPT TWO},', 'which is:',  f'{0.0001*leaf_count*pixel_cm_ratio:.3f}', 'm\N{SUPERSCRIPT TWO}')
