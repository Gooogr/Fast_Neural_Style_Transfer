import streamlit as st
import cv2
from PIL import Image
from model_functions import predict

# ~ import time

DEMO_IMG = './streamlit_imgs/demo.jpg'

st.title("Fast Neural Style Tranfer")

# Get content image from file_uploader
img_file_buffer = st.sidebar.file_uploader("Select content image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
	image = Image.open(img_file_buffer)
else:
	image = Image.open(DEMO_IMG)
content_img = st.image(image) # How to format image for screen ratio and width?

# Select model weights
style_name = st.sidebar.selectbox("Select style image", ("Van Gogh", 
														 "Wassily Kandinsky", 
														 "Sketching"))

def get_weights(style_name):
	pass

weights = get_weights(style_name)

# Create options dictionary for predict function
options = dict()
options['img_path'] = content_img
options['weights_path'] = weights
options['result_dir'] = None

# Predict and display result
predict_img = predict(options)
    
## Multile images along  one side
## https://discuss.streamlit.io/t/multiple-images-along-the-same-row/118/3
    
## Waiting text    
# ~ with st.spinner('Wait for it...'):
	# ~ time.sleep(5)
# ~ st.success('Done!')
