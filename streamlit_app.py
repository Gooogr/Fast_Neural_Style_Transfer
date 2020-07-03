import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from model_functions import predict


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
	
print(content_image.shape)

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
st.image([content_image, style_img_path],  
		 caption = ['Content Image', 'Style Image'], 
		 width = 400)

# Create options dictionary for predict function
options = dict()
options['img_path'] = content_image
options['weights_path'] = weights_path
options['result_dir'] = None

# Predict and display result
predicted_img = predict(options, write_result=False)
st.write("Result image")
st.image(predicted_img[:, :, ::-1], width=600)


# Download results
result = Image.fromarray(predicted_img)

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href

st.markdown(get_image_download_link(result), unsafe_allow_html=True)



