import streamlit as st
from model_functions import predict

# ~ import time

st.title("Fast Neural Style Tranfer")
img_file_buffer = st.sidebar.file_uploader("Select content image", type=["png", "jpg", "jpeg"])

style_name = st.sidebar.selectbox("Select style image", ("Option1", "Option2", "Option3"))

# ~ if img_file_buffer is not None:
    # ~ image = np.array(Image.open(img_file_buffer))

# ~ else:
    # ~ demo_image = DEMO_IMAGE
    # ~ image = np.array(Image.open(demo_image))
    
## Multile images along  one side
## https://discuss.streamlit.io/t/multiple-images-along-the-same-row/118/3
    
## Waiting text    
# ~ with st.spinner('Wait for it...'):
	# ~ time.sleep(5)
# ~ st.success('Done!')
