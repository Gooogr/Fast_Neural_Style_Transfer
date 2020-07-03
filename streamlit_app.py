import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model_functions import predict

# ~ import time

DEMO_IMG = './streamlit_imgs/demo.jpg'

VAN_GOGH_WEIGHT = './saved_weights/fst_night_512_weights.h5'
VAN_GOGH_IMG = './streamlit_imgs/night.jpg'

KANDINSKIY_WEIGHT = './saved_weights/fst_kandinskiy_512_weights.h5'
KANDINSKIY_IMG = './streamlit_imgs/kandinskiy.jpg'

SKETCH_WEIGHT = './saved_weights/fst_draft_512_weights.h5'
SKETCH_IMG = './streamlit_imgs/draft.jpg'

st.title("Fast Neural Style Tranfer")

# Get content image from file_uploader
img_file_buffer = st.sidebar.file_uploader("Select content image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
	content_image = np.array(Image.open(img_file_buffer))
else:
	content_image = np.array(Image.open(DEMO_IMG))

# Select model weights
style_name = st.sidebar.selectbox("Select style image", ("Van Gogh", 
														 "Wassily Kandinsky", 
														 "Sketching"))
style_dict = dict()
style_dict['Van Gogh'] = VAN_GOGH_WEIGHT, VAN_GOGH_IMG
style_dict['Wassily Kandinsky'] = KANDINSKIY_WEIGHT, KANDINSKIY_IMG
style_dict['Sketching'] = SKETCH_WEIGHT, SKETCH_IMG

weights_path, style_img_path = style_dict[style_name]

# Draw content and style images
st.image([content_image, style_img_path]) # How to format image for screen ratio and width?

# Create options dictionary for predict function
options = dict()
options['img_path'] = content_image
options['weights_path'] = weights_path
options['result_dir'] = None

# Predict and display result
predict_img = predict(options, write_result=False)
st.image(predict_img[:, :, ::-1])


if st.button('Download result'):
	st.write('Not yet :D')

    
## Multile images along  one side
## https://discuss.streamlit.io/t/multiple-images-along-the-same-row/118/3
    
## Waiting text    
# ~ with st.spinner('Wait for it...'):
	# ~ time.sleep(5)
# ~ st.success('Done!')
