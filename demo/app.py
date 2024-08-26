import streamlit as st
import requests
import base64
import json

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def run_inference(image_file):
    api_url = "http://localhost:3000/tag_image_file"
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    
    response = requests.post(api_url, files=files)
    
    return response.json()

def main():
    st.title("Image Inference Demo")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Run inference button
        if st.button('Run Inference'):
            with st.spinner('Running inference...'):
                try:
                    result = run_inference(uploaded_file)
                    st.success('Inference complete!')
                    st.json(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()