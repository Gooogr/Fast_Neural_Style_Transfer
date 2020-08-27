import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from model_functions import predict

# Constants
DEMO_IMG = './streamlit_imgs/demo.jpg'

VAN_GOGH_WEIGHT = './saved_weights/fst_night_512_weights.h5'
VAN_GOGH_IMG = './streamlit_imgs/night.jpg'

KANDINSKIY_WEIGHT = './saved_weights/fst_kandinskiy_512_weights.h5'
KANDINSKIY_IMG = './streamlit_imgs/kandinskiy.jpg'

SKETCH_WEIGHT = './saved_weights/fst_draft_512_weights.h5'
SKETCH_IMG = './streamlit_imgs/draft.jpg'

SCALED_CONTENT_IMG_WIDTH = 400

# Set site title
st.title("Fast Neural Style Tranfer")

# Get content image from file_uploader
img_file_buffer = st.sidebar.file_uploader("Select content image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
	content_image = np.array(Image.open(img_file_buffer))
else:
	content_image = np.array(Image.open(DEMO_IMG))
	
print('---Image shapes---')	
print('Content image shape:', content_image.shape)
print()

# Select model weights and style image
style_name = st.sidebar.selectbox("Select style image", ("Van Gogh", 
														 "Wassily Kandinsky", 
														 "Sketching"))
style_dict = dict()
style_dict['Van Gogh'] = VAN_GOGH_WEIGHT, VAN_GOGH_IMG
style_dict['Wassily Kandinsky'] = KANDINSKIY_WEIGHT, KANDINSKIY_IMG
style_dict['Sketching'] = SKETCH_WEIGHT, SKETCH_IMG

weights_path, style_img_path = style_dict[style_name]

# Scale style and content images. 
# I want to allign them by height, so width parameter in st.image set to None
content_scale = SCALED_CONTENT_IMG_WIDTH / content_image.shape[1]
content_scaled_height = int(content_image.shape[0] * content_scale)
content_image = cv2.resize(content_image, 
				(SCALED_CONTENT_IMG_WIDTH, content_scaled_height))
	
style_image = cv2.imread(style_img_path)
style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)	
style_scale =  content_image.shape[0] / style_image.shape[0]
style_scaled_width =  int(style_image.shape[1] * style_scale)
style_image = cv2.resize(style_image, 
				(style_scaled_width, content_scaled_height))

# Draw content and style images
st.image([content_image, style_image],  
		 caption=['Content Image', 'Style Image'], 
		 width=None)

# Create options dictionary for predict function
options = dict()
options['img_path'] = content_image
options['weights_path'] = weights_path
options['result_dir'] = None

# Predict and display result
predicted_img = predict(options, write_result=False)
st.write("Result image")
st.image(predicted_img[:, :, ::-1], width=500)


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
	href = '<a href="data:file/jpg;base64,{}">Download result</a>'.format(img_str)
	return href
	
st.markdown(get_image_download_link(result), unsafe_allow_html=True)



