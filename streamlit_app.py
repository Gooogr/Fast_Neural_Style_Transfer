import streamlit as st
from model_functions import predict

st.title("Fast Neural Style Tranfer")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# ~ if img_file_buffer is not None:
    # ~ image = np.array(Image.open(img_file_buffer))

# ~ else:
    # ~ demo_image = DEMO_IMAGE
    # ~ image = np.array(Image.open(demo_image))
