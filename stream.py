import streamlit as st
import requests

from PIL import Image
from io import BytesIO
# Streamlit app title
st.title("Image Captioning")

# Upload image
uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png'])
# Input image URL
# image_url = st.text_input("Enter Image URL")

# # Check if image URL is provided
# if image_url:
#     try:
#         # Fetch image from URL
#         response = requests.get(image_url)
#         image = Image.open(BytesIO(response.content))
        
#         # Resize the image
#         resized_image = image.resize((300, 300))
        
#         # Display the resized image
#         st.image(resized_image, caption="Image from URL", use_column_width=True)
#         files = {'image': uploaded_image.getvalue()}
#         api_url = 'http://localhost:5000/image_size'
#         response = requests.post(api_url, files=files)

#     # Check if API call was successful
#         if response.status_code == 200:
#             data = response.json()
#             width = data['width']
#             height = data['height']
#             caption = data['cap']
        
#         # Display image size and caption
#             st.write(f"Image Size: {width} x {height}")
#             st.write(f"Caption: {caption}")
#         else:
#             st.error("Error occurred while processing the image. Please try again.")
#     except Exception as e:
#         st.error("Error loading image from URL. Please try again.")

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Call the Flask API to get image size and caption
    files = {'image': uploaded_image.getvalue()}
    api_url = 'http://localhost:5000/image_size'
    response = requests.post(api_url, files=files)

    # Check if API call was successful
    if response.status_code == 200:
        data = response.json()
        width = data['width']
        height = data['height']
        caption = data['cap']
        
        # Display image size and caption
        st.write(f"Image Size: {width} x {height}")
        st.write(f"Caption: {caption}")
    else:
        st.error("Error occurred while processing the image. Please try again.")
