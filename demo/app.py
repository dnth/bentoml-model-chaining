import streamlit as st
import requests
import base64
import json
import logging
from PIL import Image, ImageDraw

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def run_inference(image_file, task):
    base_url = "http://localhost"
    
    if task == "tag_image_file":
        port = 3000
    elif task == "detect_objects_file":
        port = 3001
    else:
        raise ValueError(f"Unknown task: {task}")
    
    api_url = f"{base_url}:{port}/{task}"
    
    logging.info(f"Sending request to: {api_url}")
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    
    response = requests.post(api_url, files=files)
    
    logging.info(f"Response status code: {response.status_code}")
    logging.info(f"Response content: {response.text}")
    
    return response  # Return the raw response object

def draw_bounding_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        bbox = detection['box']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{detection['name']} {detection['confidence']:.2f}"
        draw.text((x1, y1 - 10), label, fill="red")
    return image

def main():
    logging.basicConfig(level=logging.INFO)
    
    st.title("Image Inference Demo")
    
    # Create tabs
    tabs = st.tabs(["Image Tagging", "Object Detection"])
    
    # File uploader (shared between tabs)
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image in the sidebar
        st.sidebar.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Image Tagging tab
        with tabs[0]:
            st.header("Image Tagging")
            if st.button('Run Image Tagging'):
                with st.spinner('Running image tagging...'):
                    try:
                        result = run_inference(uploaded_file, "tag_image_file")
                        st.success('Image tagging complete!')
                        # Parse the JSON content from the response
                        result_json = result.json()
                        st.json(result_json)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        
        # Object Detection tab
        with tabs[1]:
            st.header("Object Detection")
            if st.button('Run Object Detection'):
                with st.spinner('Running object detection...'):
                    try:
                        result = run_inference(uploaded_file, "detect_objects_file")
                        st.success('Object detection complete!')
                        result_json = result.json()
                        st.json(result_json)
                        
                        # Plot the detection result
                        image = Image.open(uploaded_file)
                        image_with_boxes = draw_bounding_boxes(image, result_json)
                        st.image(image_with_boxes, caption='Object Detection Result', use_column_width=True)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Full error details:")
                        st.write(e)

if __name__ == "__main__":
    main()